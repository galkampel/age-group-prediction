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
    n_trials = 30
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
    return split_groups, splitter, train_index


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

    One Optuna study per component, by cross-validation on the training
    part: a `CountModel` is scored by its Poisson deviance (each cohort's
    column for Model 1, the row sums for Model 2's total), the
    `CohortProbabilityModel` by the cohort log loss. The i-th study of a
    run gets `seed + i`, so the studies of one run do not share their
    startup trials. The estimator, its feature variant and the exposure
    are settings, each a run of its own with the same seeds and folds: in
    one study a categorical LightGBM loss split the budget unevenly and
    its best trial was worse than either loss alone, and six GLM variants
    would get five trials each. NB2 tunes its dispersion and its L1
    penalty. A Gaussian loss can predict a negative count; `CountModel`
    floors its mean, so such a trial scores badly instead of stopping the
    study. The run cells sit behind buttons: opening the notebook fits
    nothing.
    """)
    return


@app.cell
def _():
    import time

    import optuna
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, PoissonRegressor

    from age_group_prediction.hyperparameter_tuning import (
        CategoricalParameter,
        CVHyperparameterEvaluator,
        FloatParameter,
        HyperparameterStudy,
        IntParameter,
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

    # Optuna logs every trial at INFO.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        COHORT_LOG_LOSS,
        COHORT_POISSON_DEVIANCE,
        CVHyperparameterEvaluator,
        CategoricalParameter,
        CohortProbabilityModel,
        CountModel,
        FloatParameter,
        HyperparameterStudy,
        IndependentCohortModels,
        IntParameter,
        LGBMClassifier,
        LGBMRegressor,
        LogisticRegression,
        NegativeBinomialRegressor,
        POISSON_DEVIANCE,
        PoissonRegressor,
        TotalTimesProbabilityModel,
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
    LGBMClassifier,
    LGBMRegressor,
    LogisticRegression,
    NegativeBinomialRegressor,
    PoissonRegressor,
    SEARCH_SPACES,
    seed,
):
    # Each template: (estimator, search space, the FEATURE_VARIANTS keys it runs on).
    # The trees take n_apartments as a feature and the offset; the GLMs take it
    # as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4), and the
    # logistic model needs the scaled plans (it does not converge on `tree`).
    # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
    # thread per fit.
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
    REGRESSORS = {
        "lgbm_poisson": (
            LGBMRegressor(
                objective="poisson",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            ),
            SEARCH_SPACES["lgbm"],
            TREE_VARIANTS,
        ),
        "lgbm_gaussian": (
            LGBMRegressor(
                objective="regression",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            ),
            SEARCH_SPACES["lgbm"],
            TREE_VARIANTS,
        ),
        "poisson_glm": (
            PoissonRegressor(max_iter=1000),
            SEARCH_SPACES["poisson_glm"],
            COUNT_GLM_VARIANTS,
        ),
        "nb2": (
            NegativeBinomialRegressor(),
            SEARCH_SPACES["nb2"],
            COUNT_GLM_VARIANTS,
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
    return CLASSIFIERS, REGRESSORS


@app.cell
def _(CLASSIFIERS, REGRESSORS):
    # Model 1, one per cohort: (regressor, feature variant, use_exposure), every
    # regressor on each of its variants, with and without the exposure. Without
    # it a GLM is size-blind, since no GLM variant lists n_apartments.
    MODEL1_SETTINGS = {
        f"{regressor}__{variant}__{'exposure' if use_exposure else 'no_exposure'}": (
            regressor,
            variant,
            use_exposure,
        )
        for regressor, (_, _, variants) in REGRESSORS.items()
        for variant in variants
        for use_exposure in (True, False)
    }
    PROBABILITY_SETTINGS = {
        f"{classifier}__{variant}": (classifier, variant)
        for classifier, (_, _, variants) in CLASSIFIERS.items()
        for variant in variants
    }
    # Model 2: (total setting, probability setting). The total is a Model 1
    # setting on the row sums; the components are fitted independently, so
    # each is tuned once and the pairs are assembled from the tuned components.
    MODEL2_SETTINGS = {
        f"{total}__{probability}": (total, probability)
        for total in MODEL1_SETTINGS
        for probability in PROBABILITY_SETTINGS
    }
    return MODEL1_SETTINGS, MODEL2_SETTINGS, PROBABILITY_SETTINGS


@app.cell
def _(CVHyperparameterEvaluator, HyperparameterStudy, time):
    def tune_component(
        model,
        parameters,
        X,
        y,
        *,
        exposure,
        metric,
        study_seed,
        cv,
        cv_groups,
        n_trials,
    ):
        """One study of ``model`` on the rows of ``X``; the result and its seconds.

        An empty ``parameters`` scores the model as it is, in one trial: more
        trials would repeat the same fit.
        """
        evaluator = CVHyperparameterEvaluator(model, parameters, cv=cv, metric=metric)
        study = HyperparameterStudy(
            seed=study_seed, n_trials=n_trials if parameters else 1
        )
        start = time.perf_counter()
        result = study.optimize(
            lambda trial: evaluator.evaluate(trial, X, y, cv_groups, exposure=exposure)
        )
        return result, time.perf_counter() - start

    return (tune_component,)


if __name__ == "__main__":
    app.run()
