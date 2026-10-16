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
    # §8.2: the plans of glm_base, named once for the variants below.
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

    # §8.2: the base of every GLM, count or probability model; n_apartments is
    # the exposure only, never a plan.
    glm_base = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
    )
    # §8.5: candidates 2-4, each changing the count model's plans once.
    count__ses_quadratic = FeatureTransformer(
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
    count__daycare_log1p = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_saturation_plan,
            school_plan,
        ),
    )
    count__room_share_x_household_size = FeatureTransformer(
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
    count__room_share_x_ses = FeatureTransformer(
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
    count__room_share_x_household_size_and_ses = FeatureTransformer(
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

    # §8.3: the probability model takes the same base (§8.2) without an
    # exposure; §8.5's candidates 5-6 each change its plans once.
    probability__daycare_log1p = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_saturation_plan,
            school_plan,
        ),
    )
    probability__room_share_x_median_age = FeatureTransformer(
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
    probability__room_share_x_daycare = FeatureTransformer(
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
    probability__room_share_x_median_age_and_daycare = FeatureTransformer(
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
    study. The run cells sit behind buttons: opening the notebook fits
    nothing.
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
    CLASSIFIERS = {
        "logistic": (
            LogisticRegression(max_iter=1000),
            SEARCH_SPACES["logistic"],
            PROBABILITY_GLM_VARIANTS,
        ),
        "lgbm": (
            LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            ),
            SEARCH_SPACES["lgbm_classifier"],
            TREE_VARIANTS,
        ),
    }
    return CLASSIFIERS, COHORTS, COUNT_FAMILIES


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
    return (MODEL1_EVALUATORS,)


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
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    CategoricalParameter,
    Counter,
    KeyedParameter,
    MODEL1_EVALUATORS,
    MODEL1_RESULTS,
    POISSON_DEVIANCE,
    X_train,
    X_val,
    Y_train,
    Y_val,
    exposure_train,
    exposure_val,
    pd,
):
    MODEL1_MODELS = {}
    _model1_rows, _model1_choice_rows, _model1_arm_rows = [], [], []
    for _family, (_result, _seconds) in MODEL1_RESULTS.items():
        _evaluator = MODEL1_EVALUATORS[_family]
        # The best combination, refit on the training part.
        _model = _evaluator.build_model(_result.best_params).fit(
            X_train, Y_train, exposure=exposure_train
        )
        MODEL1_MODELS[_family] = _model
        _pred_train = _model.predict(X_train, exposure=exposure_train)
        _pred_val = _model.predict(X_val, exposure=exposure_val)
        for _cohort in COHORTS:
            _model1_rows.append(
                {
                    "family": _family,
                    "cohort": _cohort,
                    "train": _model.evaluate(
                        Y_train[_cohort], _pred_train[_cohort], POISSON_DEVIANCE
                    ),
                    "val": _model.evaluate(
                        Y_val[_cohort], _pred_val[_cohort], POISSON_DEVIANCE
                    ),
                }
            )
            _prefix = f"cohort_models__{_cohort}__"
            _model1_choice_rows.append(
                {
                    "family": _family,
                    "cohort": _cohort,
                    **{
                        name.removeprefix(_prefix): value
                        for name, value in _result.best_params.items()
                        if name.startswith(_prefix)
                        and name.removeprefix(_prefix)
                        in (
                            "feature_transformer",
                            "use_exposure",
                            "estimator__objective",
                        )
                    },
                }
            )
        _model1_rows.append(
            {
                "family": _family,
                "cohort": "composite",
                "cv_best": _result.best_value,
                "train": _model.evaluate(Y_train, _pred_train, COHORT_POISSON_DEVIANCE),
                "val": _model.evaluate(Y_val, _pred_val, COHORT_POISSON_DEVIANCE),
                "seconds": _seconds,
            }
        )
        # Trials per arm of every categorical: a starved arm is visible before
        # a family is declared worse.
        for _parameter in _evaluator.parameters:
            if isinstance(_parameter, (CategoricalParameter, KeyedParameter)):
                _counts = Counter(
                    trial.params[_parameter.name] for trial in _result.trials
                )
                _model1_arm_rows.extend(
                    {
                        "family": _family,
                        "parameter": _parameter.name,
                        "value": value,
                        "n_trials": count,
                    }
                    for value, count in _counts.items()
                )
    model1_df = pd.DataFrame(_model1_rows)
    model1_choices_df = pd.DataFrame(_model1_choice_rows)
    model1_arms_df = pd.DataFrame(_model1_arm_rows)
    return MODEL1_MODELS, model1_arms_df, model1_choices_df, model1_df


@app.cell
def _(mo, model1_arms_df, model1_choices_df, model1_df):
    mo.vstack(
        [
            mo.md("**Model 1: cohort Poisson deviance** (cv_best: the study's value)"),
            mo.md(f"```text\n{model1_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Best combination per cohort**"),
            mo.md(f"```text\n{model1_choices_df.to_string(index=False)}\n```"),
            mo.md("**Trials per arm**"),
            mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
        ]
    )


if __name__ == "__main__":
    app.run()
