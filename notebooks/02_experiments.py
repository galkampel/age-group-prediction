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

    FEATURE_TRANSFORMERS = {
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
    return (FEATURE_TRANSFORMERS,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Hyperparameter tuning

    One Optuna study per model family and exposure (on or off, fixed for the
    study), on the composite: the model that is scored is the model that is
    tuned, by the cohort Poisson deviance of its cross-validation folds on the
    training part, the plain mean over folds. A study's fixed settings are a
    frozen dataclass, its component; a trial samples per cohort (per component
    for Model 2) the feature transformer by name and the estimator's
    hyperparameters, under the names `<component>__<parameter>`, and builds
    the estimator, its `CountModel` and the composite with constructor calls.
    LightGBM's two losses, Poisson and Gaussian, are separate families, and
    the exposure a study setting: a study that sampled the loss split 30
    trials 10/20 and did worse than either loss alone, and the exposure
    changes the target (a count or a weighted rate). `n_trials = 200`, so
    that no categorical arm starves; the table of trials per arm shows it.
    The i-th study gets `seed + i`; the studies of one model run
    in parallel with joblib. A trial whose fit raises (an NB2 fit that does not
    converge) is marked failed and the study goes on; the tables count them.
    Only each study's best run is rebuilt, refit on the training part and
    scored on training and validation. A Gaussian loss can predict a negative
    count, which `CountModel` floors at `y_min` (eps). Model 2 is tuned the
    same way, one study per total family (the list `MODEL2_TOTALS`, edited
    after Model 1's table), the total's exposure and a classifier family,
    under `total__…` and `probability__…`; the probability model ignores the
    exposure, and its calibration is searched. The last table shows each model's best family on
    validation. The run cells sit behind buttons: opening the notebook fits
    nothing.
    """)
    return


@app.cell
def _():
    import abc
    import dataclasses
    import time
    from collections import Counter
    from functools import partial
    from types import MappingProxyType
    from typing import ClassVar, Literal, get_args, get_type_hints

    import numpy as np
    import optuna
    from joblib import Parallel, delayed
    from lightgbm import LGBMClassifier, LGBMRegressor
    from optuna.distributions import CategoricalDistribution
    from optuna.samplers import TPESampler
    from optuna.trial import FixedTrial, TrialState
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.metrics import mean_poisson_deviance

    from age_group_prediction.modeling import (
        CohortProbabilityModel,
        CountModel,
        IndependentCohortModels,
        NegativeBinomialRegressor,
        TotalTimesProbabilityModel,
    )
    from age_group_prediction.scoring import Metric, cohort_poisson_deviance

    # Optuna logs every trial at INFO; the tuning function repeats this in
    # each joblib worker.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        CategoricalDistribution,
        ClassVar,
        CohortProbabilityModel,
        CountModel,
        Counter,
        FixedTrial,
        IndependentCohortModels,
        LGBMClassifier,
        LGBMRegressor,
        Literal,
        LogisticRegression,
        MappingProxyType,
        Metric,
        NegativeBinomialRegressor,
        Parallel,
        PoissonRegressor,
        TPESampler,
        TotalTimesProbabilityModel,
        TrialState,
        abc,
        cohort_poisson_deviance,
        dataclasses,
        delayed,
        get_args,
        get_type_hints,
        mean_poisson_deviance,
        np,
        optuna,
        partial,
        time,
    )


@app.cell
def _(Metric, cohort_poisson_deviance, mean_poisson_deviance):
    # The study's objective and the tables' per-cohort score, with the direction
    # Optuna is given; a loss, so "minimize".
    COHORT_POISSON_DEVIANCE = Metric(
        "cohort_poisson_deviance", cohort_poisson_deviance, direction="minimize"
    )
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance, direction="minimize"
    )
    return COHORT_POISSON_DEVIANCE, POISSON_DEVIANCE


@app.cell
def _(Literal, Y, get_args, np):
    COHORTS = list(Y.columns)
    # The smallest mean a CountModel predicts: a Gaussian loss can predict a
    # negative count, the Poisson deviance raises at 0 and stays finite at eps.
    PREDICTION_FLOOR = float(np.finfo(float).eps)
    # Plain assignments, not the `type` statement: cloudpickle cannot pickle a
    # class annotated with a `type` alias defined in a cell, and joblib's
    # workers receive the components by cloudpickle.
    CountFamily = Literal["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
    ProbabilityFamily = Literal["logistic", "lgbm"]
    COUNT_FAMILIES = get_args(CountFamily)
    PROBABILITY_FAMILIES = get_args(ProbabilityFamily)
    return (
        COHORTS,
        COUNT_FAMILIES,
        CountFamily,
        PREDICTION_FLOOR,
        PROBABILITY_FAMILIES,
        ProbabilityFamily,
    )


@app.cell
def _(dataclasses):
    @dataclasses.dataclass(frozen=True)
    class LGBMParams:
        """LightGBM's searched hyperparameters, shared by the regressor and the classifier."""

        max_depth: int
        num_leaves: int
        min_child_samples: int
        learning_rate: float
        n_estimators: int
        reg_alpha: float
        reg_lambda: float
        min_split_gain: float
        subsample: float
        colsample_bytree: float

        @classmethod
        def suggest(cls, trial, prefix, **fixed):
            return cls(
                max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
                num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
                min_child_samples=trial.suggest_int(
                    f"{prefix}min_child_samples", 5, 40
                ),
                learning_rate=trial.suggest_float(
                    f"{prefix}learning_rate", 0.01, 0.2, log=True
                ),
                n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
                reg_alpha=trial.suggest_float(
                    f"{prefix}reg_alpha", 1e-8, 10.0, log=True
                ),
                reg_lambda=trial.suggest_float(
                    f"{prefix}reg_lambda", 1e-8, 10.0, log=True
                ),
                min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
                subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float(
                    f"{prefix}colsample_bytree", 0.7, 1.0
                ),
                **fixed,
            )

    @dataclasses.dataclass(frozen=True)
    class LGBMRegressorParams(LGBMParams):
        # LightGBM's name for the loss, "poisson" or "regression" (Gaussian):
        # fixed by the family and recorded, not sampled; `suggest` takes it as
        # a keyword, `suggest(trial, prefix, objective="poisson")`.
        objective: str

    @dataclasses.dataclass(frozen=True)
    class LGBMClassifierParams(LGBMParams):
        """The classifier's; its objective, multiclass, is not searched."""

    @dataclasses.dataclass(frozen=True)
    class PoissonGLMParams:
        alpha: float

        @classmethod
        def suggest(cls, trial, prefix):
            return cls(
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
            )

    @dataclasses.dataclass(frozen=True)
    class NB2Params:
        # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
        # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
        dispersion: float
        alpha: float

        @classmethod
        def suggest(cls, trial, prefix):
            return cls(
                dispersion=trial.suggest_float(
                    f"{prefix}dispersion", 0.01, 1.0, log=True
                ),
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
            )

    @dataclasses.dataclass(frozen=True)
    class LogisticParams:
        C: float

        @classmethod
        def suggest(cls, trial, prefix):
            return cls(C=trial.suggest_float(f"{prefix}C", 1e-3, 100.0, log=True))

    return (
        LGBMClassifierParams,
        LGBMParams,
        LGBMRegressorParams,
        LogisticParams,
        NB2Params,
        PoissonGLMParams,
    )


@app.cell
def _(
    ClassVar,
    CountFamily,
    LGBMClassifier,
    LGBMClassifierParams,
    LGBMRegressor,
    LGBMRegressorParams,
    LogisticParams,
    LogisticRegression,
    MappingProxyType,
    NB2Params,
    NegativeBinomialRegressor,
    PoissonGLMParams,
    PoissonRegressor,
    ProbabilityFamily,
    abc,
    dataclasses,
    get_args,
    get_type_hints,
):
    # The components read no notebook global: the feature transformers come in
    # as an argument, so the classes can move to a package.
    @dataclasses.dataclass(frozen=True)
    class Component(abc.ABC):
        """A named part of a composite and its settings fixed for one study.

        A trial samples the rest under ``<name>__``: the feature transformer and
        the estimator's hyperparameters. Sampling (``suggest_params``) and
        construction (``make_estimator``) are separate steps, so the sampled
        params are a record of their own; the model and the composite are built
        outside.
        """

        name: str
        family: str
        # Set by each subclass: each family's candidate feature transformers, by
        # name. Fixed on the class, so no component carries another family's.
        FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]]

        def __post_init__(self):
            # At construction, before the parallel studies start. The subclass
            # annotates `family` with its Literal, which holds the valid names.
            families = get_args(get_type_hints(type(self))["family"])
            if self.family not in families:
                raise ValueError(
                    f"unknown {type(self).__name__} family {self.family!r}"
                )
            if self.family not in self.FEATURE_TRANSFORMER_NAMES:
                raise ValueError(f"no feature transformers for family {self.family!r}")

        @property
        def prefix(self):
            """The trial's parameter names are ``<name>__<parameter>``."""
            return f"{self.name}__"

        def suggest_feature_transformer(self, trial, feature_transformers):
            """The feature transformer whose name ``trial`` samples from the family's.

            The trial samples a name, since Optuna's categorical choices must be
            primitives; ``feature_transformers`` maps a name to its transformer.
            """
            name = trial.suggest_categorical(
                f"{self.prefix}feature_transformer",
                list(self.FEATURE_TRANSFORMER_NAMES[self.family]),
            )
            return feature_transformers[name]

        @abc.abstractmethod
        def suggest_params(self, trial):
            """The estimator's hyperparameters, sampled under ``prefix``."""

        @abc.abstractmethod
        def make_estimator(self, hyperparameters, *, seed):
            """The unfitted estimator: the family's fixed settings and ``hyperparameters``.

            ``hyperparameters`` maps the estimator's keywords to their values.
            """

    @dataclasses.dataclass(frozen=True)
    class CountComponent(Component):
        """A CountModel's settings fixed for one study: the family and the exposure."""

        family: CountFamily
        use_exposure: bool
        # The trees take n_apartments as a feature and the offset; the GLMs take
        # it as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4).
        FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
            MappingProxyType(
                {
                    **dict.fromkeys(
                        ("lgbm_poisson", "lgbm_gaussian"),
                        ("tree", "tree__ses_quadratic"),
                    ),
                    **dict.fromkeys(
                        ("poisson_glm", "nb2"),
                        (
                            "glm_base",
                            "count__ses_quadratic",
                            "count__daycare_log1p",
                            "count__room_share_x_household_size",
                            "count__room_share_x_ses",
                            "count__room_share_x_household_size_and_ses",
                        ),
                    ),
                }
            )
        )

        def suggest_params(self, trial):
            if self.family == "lgbm_poisson":
                return LGBMRegressorParams.suggest(
                    trial, self.prefix, objective="poisson"
                )
            elif self.family == "lgbm_gaussian":
                return LGBMRegressorParams.suggest(
                    trial, self.prefix, objective="regression"
                )
            elif self.family == "poisson_glm":
                return PoissonGLMParams.suggest(trial, self.prefix)
            elif self.family == "nb2":
                return NB2Params.suggest(trial, self.prefix)
            else:
                raise ValueError(f"unknown count family {self.family!r}")

        def make_estimator(self, hyperparameters, *, seed):
            if self.family in ("lgbm_poisson", "lgbm_gaussian"):
                # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
                # thread per fit, so the parallel studies do not oversubscribe the cores.
                return LGBMRegressor(
                    subsample_freq=1,
                    n_jobs=1,
                    verbosity=-1,
                    random_state=seed,
                    **hyperparameters,
                )
            elif self.family == "poisson_glm":
                return PoissonRegressor(max_iter=1000, **hyperparameters)
            elif self.family == "nb2":
                return NegativeBinomialRegressor(**hyperparameters)
            else:
                raise ValueError(f"unknown count family {self.family!r}")

    @dataclasses.dataclass(frozen=True)
    class ProbabilityComponent(Component):
        """Model 2's CohortProbabilityModel settings fixed for one study; the exposure does not apply."""

        family: ProbabilityFamily
        # The logistic model needs the scaled plans (it does not converge on `tree`).
        FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
            MappingProxyType(
                {
                    "logistic": (
                        "glm_base",
                        "probability__daycare_log1p",
                        "probability__room_share_x_median_age",
                        "probability__room_share_x_daycare",
                        "probability__room_share_x_median_age_and_daycare",
                    ),
                    "lgbm": ("tree", "tree__ses_quadratic"),
                }
            )
        )

        def suggest_params(self, trial):
            if self.family == "lgbm":
                return LGBMClassifierParams.suggest(trial, self.prefix)
            elif self.family == "logistic":
                return LogisticParams.suggest(trial, self.prefix)
            else:
                raise ValueError(f"unknown probability family {self.family!r}")

        def make_estimator(self, hyperparameters, *, seed):
            if self.family == "lgbm":
                return LGBMClassifier(
                    objective="multiclass",
                    subsample_freq=1,
                    n_jobs=1,
                    verbosity=-1,
                    random_state=seed,
                    **hyperparameters,
                )
            elif self.family == "logistic":
                return LogisticRegression(max_iter=1000, **hyperparameters)
            else:
                raise ValueError(f"unknown probability family {self.family!r}")

        def suggest_calibration_method(self, trial):
            # No isotonic: its lowest score block can hold no child of a cohort and
            # then maps a building there to probability 0, an infinite log loss
            # (measured 2026-10-15 with the LightGBM classifier).
            return trial.suggest_categorical(
                f"{self.prefix}calibration_method", [None, "temperature", "sigmoid"]
            )

    return Component, CountComponent, ProbabilityComponent


@app.cell
def _(
    CohortProbabilityModel,
    CountModel,
    IndependentCohortModels,
    TotalTimesProbabilityModel,
    dataclasses,
):
    def suggest_count_model(trial, component, *, feature_transformers, y_min, seed):
        """A CountModel whose feature transformer and hyperparameters ``trial`` samples under the component's name."""
        hyperparameters = dataclasses.asdict(component.suggest_params(trial))
        # CountModel clones the transformer at fit, so the shared declaration is safe.
        return CountModel(
            estimator=component.make_estimator(hyperparameters, seed=seed),
            use_exposure=component.use_exposure,
            feature_transformer=component.suggest_feature_transformer(
                trial, feature_transformers
            ),
            y_min=y_min,
        )

    def suggest_model1(trial, components, *, feature_transformers, y_min, seed):
        """Model 1: one CountModel per component, keyed by the component's name, a cohort."""
        return IndependentCohortModels(
            {
                component.name: suggest_count_model(
                    trial,
                    component,
                    feature_transformers=feature_transformers,
                    y_min=y_min,
                    seed=seed,
                )
                for component in components
            }
        )

    def suggest_model2(trial, total, probability, *, feature_transformers, y_min, seed):
        """Model 2: the total's CountModel and the classifier, each under its component's name."""
        total_model = suggest_count_model(
            trial,
            total,
            feature_transformers=feature_transformers,
            y_min=y_min,
            seed=seed,
        )
        hyperparameters = dataclasses.asdict(probability.suggest_params(trial))
        return TotalTimesProbabilityModel(
            total_model=total_model,
            probability_model=CohortProbabilityModel(
                estimator=probability.make_estimator(hyperparameters, seed=seed),
                calibration_method=probability.suggest_calibration_method(trial),
                feature_transformer=probability.suggest_feature_transformer(
                    trial, feature_transformers
                ),
            ),
        )

    return suggest_count_model, suggest_model1, suggest_model2


if __name__ == "__main__":
    app.run()
