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

    One run per model family and exposure (on or off, fixed for the run), each
    in its own cell: one Optuna study on the composite, so the model that is
    scored is the model that is tuned, by the cohort Poisson deviance of its
    cross-validation folds on the training part, the plain mean over folds.
    A trial draws, per cohort (per part for Model 2), the estimator's
    hyperparameters and the feature transformer by name, under the names
    `<component>__<parameter>`; they make a component, a complete record of
    that part, and the composite is built from the components with
    constructor calls. LightGBM's two losses, Poisson and Gaussian, are
    separate families, and the exposure a run setting: a study that sampled
    the loss split 30 trials 10/20 and did worse than either loss alone, and
    the exposure changes the target (a count or a weighted rate).
    `n_trials = 200`; the table of trials per feature transformer shows
    whether any was tried too rarely to be judged. The runs go one after
    another, run i seeded `seed + i`, and each trial fits its 5 folds in
    parallel. A trial whose fit raises (an NB2 fit that does not converge) is
    marked failed and the run goes on; the tables count them. Each run keeps
    every trial and its best components, which alone are rebuilt, refit on
    the training part and scored on training and validation. A Gaussian loss
    can predict a negative count, which `CountModel` floors at `y_min` (eps).
    Model 2 is tuned the same way, one run per total family (the list
    `MODEL2_TOTALS`, edited after Model 1's table), the total's exposure and
    a classifier family, under `total__…` and `probability__…`; the
    probability model ignores the exposure, and its calibration is searched.
    The last table shows each model's best family on validation. The run
    cells sit behind buttons: opening the notebook fits nothing.
    """)
    return


@app.cell
def _():
    import abc
    import dataclasses
    import time
    from functools import partial
    from types import MappingProxyType
    from typing import ClassVar, Literal, get_args, get_type_hints

    import numpy as np
    import optuna
    from joblib import Parallel, delayed
    from lightgbm import LGBMClassifier, LGBMRegressor
    from optuna.samplers import TPESampler
    from optuna.trial import FixedTrial
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.metrics import mean_poisson_deviance

    from age_group_prediction.aggregation import Mean
    from age_group_prediction.modeling import (
        CohortProbabilityModel,
        CountModel,
        IndependentCohortModels,
        NegativeBinomialRegressor,
        TotalTimesProbabilityModel,
    )
    from age_group_prediction.scoring import Metric, cohort_poisson_deviance

    # Optuna logs every trial at INFO; the studies run in this process.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        ClassVar,
        CohortProbabilityModel,
        CountModel,
        FixedTrial,
        IndependentCohortModels,
        LGBMClassifier,
        LGBMRegressor,
        Literal,
        LogisticRegression,
        MappingProxyType,
        Mean,
        Metric,
        NegativeBinomialRegressor,
        Parallel,
        PoissonRegressor,
        TPESampler,
        TotalTimesProbabilityModel,
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
    PROBABILITY_FAMILIES = get_args(ProbabilityFamily)
    return (
        COHORTS,
        CountFamily,
        PREDICTION_FLOOR,
        PROBABILITY_FAMILIES,
        ProbabilityFamily,
    )


@app.cell
def _(dataclasses):
    # Hyperparameters only, no Optuna: the field names are the estimator's
    # keywords, so a component makes its estimator as
    # Estimator(<fixed settings>, **asdict(params)).
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class LGBMParams:
        """LightGBM's searched hyperparameters, for the regressor and the classifier alike."""

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

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class PoissonGLMParams:
        alpha: float

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class NB2Params:
        dispersion: float
        alpha: float

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class LogisticParams:
        C: float

    return LGBMParams, LogisticParams, NB2Params, PoissonGLMParams


@app.cell
def _(
    CountFamily,
    LGBMClassifier,
    LGBMParams,
    LGBMRegressor,
    LogisticParams,
    LogisticRegression,
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
    # Complete records of one part of a run, built by hand or from a trial's
    # draws; no Optuna here (D10, D11). They read no notebook global, so the
    # classes can move to a package.
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Component(abc.ABC):
        """A named part of a composite: its family, feature transformer and hyperparameters."""

        name: str
        family: str
        # A key of FEATURE_TRANSFORMERS, resolved by the builder, so the record
        # stays JSON.
        feature_transformer_name: str

        def __post_init__(self):
            # The subclass annotates `family` with its Literal, which holds the
            # valid names.
            families = get_args(get_type_hints(type(self))["family"])
            if self.family not in families:
                raise ValueError(
                    f"unknown {type(self).__name__} family {self.family!r}"
                )

        @abc.abstractmethod
        def make_estimator(self, *, seed):
            """The unfitted estimator: the family's fixed settings and ``hyperparameters``."""

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class CountComponent(Component):
        """A CountModel's part of a run; ``use_exposure`` makes the target a weighted rate."""

        family: CountFamily
        use_exposure: bool
        hyperparameters: LGBMParams | PoissonGLMParams | NB2Params

        def make_estimator(self, *, seed):
            hyperparameters = dataclasses.asdict(self.hyperparameters)
            if self.family in ("lgbm_poisson", "lgbm_gaussian"):
                # LightGBM's name for the loss, which the family fixes (D6).
                objective = "poisson" if self.family == "lgbm_poisson" else "regression"
                # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
                # thread per fit, so the parallel folds do not oversubscribe the cores.
                return LGBMRegressor(
                    objective=objective,
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

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ProbabilityComponent(Component):
        """Model 2's CohortProbabilityModel part of a run; the exposure does not apply."""

        family: ProbabilityFamily
        calibration_method: str | None
        hyperparameters: LGBMParams | LogisticParams

        def make_estimator(self, *, seed):
            hyperparameters = dataclasses.asdict(self.hyperparameters)
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

    return Component, CountComponent, ProbabilityComponent


@app.cell
def _(
    ClassVar,
    CountFamily,
    LGBMParams,
    LogisticParams,
    MappingProxyType,
    NB2Params,
    PoissonGLMParams,
    ProbabilityFamily,
    abc,
    dataclasses,
    get_args,
    get_type_hints,
):
    # Every Optuna call (D10, D11): a search space, given a component's name and
    # family, samples its choices from a trial under `<name>__<parameter>`; the
    # caller feeds them into the component.
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class SearchSpace(abc.ABC):
        """The search of one component's choices, given its name and family."""

        name: str
        family: str
        # Set by each subclass: each family's candidate feature transformers, by
        # name. Fixed on the class, so no search space carries another family's.
        FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]]

        def __post_init__(self):
            # At construction, before a run starts.
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

        def suggest_feature_transformer_name(self, trial):
            """A name from the family's candidates; Optuna's choices must be primitives."""
            return trial.suggest_categorical(
                f"{self.prefix}feature_transformer",
                list(self.FEATURE_TRANSFORMER_NAMES[self.family]),
            )

        @abc.abstractmethod
        def suggest_hyperparameters(self, trial):
            """The family's params record, sampled under ``prefix``."""

        def _suggest_lgbm(self, trial):
            prefix = self.prefix
            return LGBMParams(
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
            )

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class CountSearchSpace(SearchSpace):
        """A count component's search: its hyperparameters and feature transformer."""

        family: CountFamily
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

        def suggest_hyperparameters(self, trial):
            prefix = self.prefix
            if self.family in ("lgbm_poisson", "lgbm_gaussian"):
                return self._suggest_lgbm(trial)
            elif self.family == "poisson_glm":
                return PoissonGLMParams(
                    alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
                )
            elif self.family == "nb2":
                # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
                # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
                return NB2Params(
                    dispersion=trial.suggest_float(
                        f"{prefix}dispersion", 0.01, 1.0, log=True
                    ),
                    alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
                )
            else:
                raise ValueError(f"unknown count family {self.family!r}")

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ProbabilitySearchSpace(SearchSpace):
        """A probability component's search: its hyperparameters, calibration and feature transformer."""

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

        def suggest_hyperparameters(self, trial):
            if self.family == "lgbm":
                return self._suggest_lgbm(trial)
            elif self.family == "logistic":
                return LogisticParams(
                    C=trial.suggest_float(f"{self.prefix}C", 1e-3, 100.0, log=True)
                )
            else:
                raise ValueError(f"unknown probability family {self.family!r}")

        def suggest_calibration_method(self, trial):
            # No isotonic: its lowest score block can hold no child of a cohort and
            # then maps a building there to probability 0, an infinite log loss
            # (measured 2026-10-15 with the LightGBM classifier).
            return trial.suggest_categorical(
                f"{self.prefix}calibration_method", [None, "temperature", "sigmoid"]
            )

    return CountSearchSpace, ProbabilitySearchSpace, SearchSpace


@app.cell
def _(
    CohortProbabilityModel,
    CountComponent,
    CountModel,
    CountSearchSpace,
    IndependentCohortModels,
    ProbabilityComponent,
    ProbabilitySearchSpace,
    TotalTimesProbabilityModel,
):
    # The suggest functions call a search space's methods explicitly and feed the
    # draws into a component; the build functions turn components into models,
    # with no Optuna. A study is one family combination and exposure (D12): the
    # objective calls suggest_model1/2, and the best run's refit rebuilds the best
    # components with suggest_model1/2_components on FixedTrial(best_params).
    def suggest_count_component(trial, space, *, use_exposure):
        """The count component of ``trial``'s draws in ``space``; the exposure is the study's."""
        # Keyword arguments are evaluated left to right, so this is the order of
        # the draws: the hyperparameters, then the transformer. A seeded study's
        # result depends on it.
        return CountComponent(
            name=space.name,
            family=space.family,
            use_exposure=use_exposure,
            hyperparameters=space.suggest_hyperparameters(trial),
            feature_transformer_name=space.suggest_feature_transformer_name(trial),
        )

    def suggest_probability_component(trial, space):
        """The probability component of ``trial``'s draws in ``space``."""
        # The draws, in order: the hyperparameters, the calibration, the transformer.
        return ProbabilityComponent(
            name=space.name,
            family=space.family,
            hyperparameters=space.suggest_hyperparameters(trial),
            calibration_method=space.suggest_calibration_method(trial),
            feature_transformer_name=space.suggest_feature_transformer_name(trial),
        )

    def suggest_model1_components(trial, *, family, use_exposure, cohorts):
        """Model 1's components ``trial`` draws: one per cohort, all of ``family``."""
        return tuple(
            suggest_count_component(
                trial,
                CountSearchSpace(name=cohort, family=family),
                use_exposure=use_exposure,
            )
            for cohort in cohorts
        )

    def suggest_model2_components(
        trial, *, total_family, probability_family, use_exposure
    ):
        """Model 2's total and probability components ``trial`` draws, the total first."""
        total = suggest_count_component(
            trial,
            CountSearchSpace(name="total", family=total_family),
            use_exposure=use_exposure,
        )
        probability = suggest_probability_component(
            trial, ProbabilitySearchSpace(name="probability", family=probability_family)
        )
        return total, probability

    def make_count_model(component, *, feature_transformer_dict, y_min, seed):
        """The CountModel of a count component."""
        # CountModel clones the transformer at fit, so the shared declaration is safe.
        return CountModel(
            estimator=component.make_estimator(seed=seed),
            use_exposure=component.use_exposure,
            feature_transformer=feature_transformer_dict[
                component.feature_transformer_name
            ],
            y_min=y_min,
        )

    def build_model1(components, *, feature_transformer_dict, y_min, seed):
        """Model 1: one CountModel per component, keyed by the component's name, a cohort."""
        return IndependentCohortModels(
            {
                component.name: make_count_model(
                    component,
                    feature_transformer_dict=feature_transformer_dict,
                    y_min=y_min,
                    seed=seed,
                )
                for component in components
            }
        )

    def build_model2(total, probability, *, feature_transformer_dict, y_min, seed):
        """Model 2: the total's CountModel and the cohort probabilities' classifier."""
        return TotalTimesProbabilityModel(
            total_model=make_count_model(
                total,
                feature_transformer_dict=feature_transformer_dict,
                y_min=y_min,
                seed=seed,
            ),
            probability_model=CohortProbabilityModel(
                estimator=probability.make_estimator(seed=seed),
                calibration_method=probability.calibration_method,
                feature_transformer=feature_transformer_dict[
                    probability.feature_transformer_name
                ],
            ),
        )

    def suggest_model1(
        trial, *, family, use_exposure, cohorts, feature_transformer_dict, y_min, seed
    ):
        """Model 1 of the components ``trial`` draws for one family and exposure."""
        components = suggest_model1_components(
            trial, family=family, use_exposure=use_exposure, cohorts=cohorts
        )
        return build_model1(
            components,
            feature_transformer_dict=feature_transformer_dict,
            y_min=y_min,
            seed=seed,
        )

    def suggest_model2(
        trial,
        *,
        total_family,
        probability_family,
        use_exposure,
        feature_transformer_dict,
        y_min,
        seed,
    ):
        """Model 2 of the components ``trial`` draws for one family pair and exposure."""
        total, probability = suggest_model2_components(
            trial,
            total_family=total_family,
            probability_family=probability_family,
            use_exposure=use_exposure,
        )
        return build_model2(
            total,
            probability,
            feature_transformer_dict=feature_transformer_dict,
            y_min=y_min,
            seed=seed,
        )

    return (
        build_model1,
        build_model2,
        make_count_model,
        suggest_count_component,
        suggest_model1,
        suggest_model1_components,
        suggest_model2,
        suggest_model2_components,
        suggest_probability_component,
    )


@app.cell
def _(Mean, Parallel, TPESampler, delayed, optuna, time):
    def fold_score(model, X, Y, exposure, train_index, val_index, metric):
        """The score on one fold's validation rows of ``model`` fitted on its training rows."""
        X_train, Y_train, exposure_train = (
            X.iloc[train_index],
            Y.iloc[train_index],
            exposure.iloc[train_index],
        )
        X_val, Y_val, exposure_val = (
            X.iloc[val_index],
            Y.iloc[val_index],
            exposure.iloc[val_index],
        )
        model.fit(X_train, Y_train, exposure=exposure_train)
        Y_val_pred = model.predict(X_val, exposure=exposure_val)
        return model.evaluate(Y_val, Y_val_pred, metric)

    def cv_score(
        model, X, Y, exposure, cv, cv_groups, metric, aggregator=None, n_jobs=1
    ):
        """The aggregated fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state).

        ``aggregator`` combines the fold scores; None means the plain mean.
        ``n_jobs`` processes fit the folds in parallel; the scores are the
        same as one after another, so a seeded study is unchanged. The
        processes do not inherit a script's ``-W`` filters: a warning in a fold
        raises there only under the environment's ``PYTHONWARNINGS`` (D15).
        """
        # The aggregator decides the selection (D9); the plain mean by default,
        # since the folds differ by one row at most.
        aggregator = Mean() if aggregator is None else aggregator
        folds = list(cv.split(X, Y, cv_groups))
        scores = Parallel(n_jobs=n_jobs)(
            delayed(fold_score)(model, X, Y, exposure, train_index, val_index, metric)
            for train_index, val_index in folds
        )
        fold_sizes = [len(val_index) for _, val_index in folds]
        return aggregator.aggregate(scores, fold_sizes)

    def tune_model(
        suggest_model,
        X,
        Y,
        exposure,
        cv,
        cv_groups,
        metric,
        *,
        study_seed,
        n_trials,
        aggregator=None,
        n_jobs=1,
    ):
        """One study of the model ``suggest_model(trial)`` builds, on the rows of ``X``; the study and its seconds.

        A trial whose fit raises RuntimeError (an NB2 fit that does not
        converge, re-raised as such from a fold's process) is marked FAIL and
        the study goes on; without ``catch`` one such trial would end the
        study. ``aggregator`` and ``n_jobs`` go to ``cv_score``.
        """

        def objective(trial):
            # Optuna passes the trial alone; the data is bound here, not to a cell.
            return cv_score(
                suggest_model(trial),
                X,
                Y,
                exposure,
                cv,
                cv_groups,
                metric,
                aggregator,
                n_jobs,
            )

        study = optuna.create_study(
            direction=metric.direction, sampler=TPESampler(seed=study_seed)
        )
        start = time.perf_counter()
        study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
        return study, time.perf_counter() - start

    return cv_score, fold_score, tune_model


@app.cell
def _(mo):
    run_model1 = mo.ui.run_button(label="Tune Model 1")
    mo.md(
        f"{run_model1} 8 runs, one cell each (a count family × exposure on or off); "
        "each trial fits its folds in parallel."
    )
    return (run_model1,)


@app.cell
def _():
    def fit_and_score(model, train_age_group, val_age_group, *, cohort_metric, metric):
        """Fit ``model`` on the training part; it and its score table on training and validation.

        A row per cohort column, scored by ``cohort_metric``, then the
        composite, scored by ``metric``.
        """
        model.fit(
            train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure
        )
        y_pred_train = model.predict(
            train_age_group.X, exposure=train_age_group.exposure
        )
        y_pred_val = model.predict(val_age_group.X, exposure=val_age_group.exposure)
        score_table = [
            {
                "cohort": cohort,
                "train": model.evaluate(
                    train_age_group.Y[cohort], y_pred_train[cohort], cohort_metric
                ),
                "val": model.evaluate(
                    val_age_group.Y[cohort], y_pred_val[cohort], cohort_metric
                ),
            }
            for cohort in train_age_group.Y.columns
        ]
        score_table.append(
            {
                "cohort": "composite",
                "train": model.evaluate(train_age_group.Y, y_pred_train, metric),
                "val": model.evaluate(val_age_group.Y, y_pred_val, metric),
            }
        )
        return model, score_table

    return (fit_and_score,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 1 of 8: LightGBM with the Poisson loss, exposure on.
    _family, _use_exposure = "lgbm_poisson", True
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 0,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_lgbm_poisson_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_lgbm_poisson_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 2 of 8: LightGBM with the Poisson loss, exposure off.
    _family, _use_exposure = "lgbm_poisson", False
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 1,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_lgbm_poisson_no_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_lgbm_poisson_no_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 3 of 8: LightGBM with the Gaussian loss, exposure on.
    _family, _use_exposure = "lgbm_gaussian", True
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 2,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_lgbm_gaussian_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_lgbm_gaussian_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 4 of 8: LightGBM with the Gaussian loss, exposure off.
    _family, _use_exposure = "lgbm_gaussian", False
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 3,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_lgbm_gaussian_no_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_lgbm_gaussian_no_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 5 of 8: the Poisson GLM, exposure on.
    _family, _use_exposure = "poisson_glm", True
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 4,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_poisson_glm_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_poisson_glm_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 6 of 8: the Poisson GLM, exposure off.
    _family, _use_exposure = "poisson_glm", False
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 5,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_poisson_glm_no_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_poisson_glm_no_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 7 of 8: the negative binomial GLM (NB2), exposure on.
    _family, _use_exposure = "nb2", True
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 6,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_nb2_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_nb2_exposure,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    FEATURE_TRANSFORMERS,
    FixedTrial,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_model1,
    cv,
    cv_groups,
    dataclasses,
    fit_and_score,
    mo,
    n_splits,
    n_trials,
    partial,
    run_model1,
    seed,
    suggest_model1,
    suggest_model1_components,
    train_age_group,
    tune_model,
    val_age_group,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # Model 1, run 8 of 8: the negative binomial GLM (NB2), exposure off.
    _family, _use_exposure = "nb2", False
    _study, _seconds = tune_model(
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + 7,
        n_trials=n_trials,
        n_jobs=n_splits,
    )
    # The best components, rebuilt from the best trial's parameters, are the
    # model scored and the record kept (D10-D12, D14).
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    _, _score_table = fit_and_score(
        build_model1(
            _best,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model1_nb2_no_exposure = {
        "family": _family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [dataclasses.asdict(c) for c in _best],
        "cv_best": _study.best_value,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_nb2_no_exposure,)


@app.cell
def _(
    CountSearchSpace,
    model1_lgbm_gaussian_exposure,
    model1_lgbm_gaussian_no_exposure,
    model1_lgbm_poisson_exposure,
    model1_lgbm_poisson_no_exposure,
    model1_nb2_exposure,
    model1_nb2_no_exposure,
    model1_poisson_glm_exposure,
    model1_poisson_glm_no_exposure,
    pd,
):
    MODEL1_RUNS = [
        model1_lgbm_poisson_exposure,
        model1_lgbm_poisson_no_exposure,
        model1_lgbm_gaussian_exposure,
        model1_lgbm_gaussian_no_exposure,
        model1_poisson_glm_exposure,
        model1_poisson_glm_no_exposure,
        model1_nb2_exposure,
        model1_nb2_no_exposure,
    ]
    _score_rows, _arm_rows = [], []
    for _run in MODEL1_RUNS:
        _settings = {"family": _run["family"], "use_exposure": _run["use_exposure"]}
        _trials = _run["trials"]
        _complete = _trials[_trials["state"] == "COMPLETE"]
        _score_rows.extend({**_settings, **row} for row in _run["score_table"][:-1])
        _score_rows.append(
            {
                **_settings,
                **_run["score_table"][-1],
                "cv_best": _run["cv_best"],
                "n_complete": len(_complete),
                "n_failed": int((_trials["state"] == "FAIL").sum()),
                "seconds": _run["seconds"],
            }
        )
        # The completed trials per feature transformer, Model 1's only categorical
        # choice: TPE concentrates on what looked good early, so a transformer
        # tried a few times has not been judged (D14). Every candidate of the
        # family is listed, so one never tried shows 0 rather than nothing.
        _candidates = CountSearchSpace.FEATURE_TRANSFORMER_NAMES[_run["family"]]
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
    return MODEL1_RUNS, model1_arms_df, model1_df


@app.cell
def _(mo, model1_arms_df, model1_df):
    mo.vstack(
        [
            mo.md(
                "**Model 1: cohort Poisson deviance** (cv_best: the run's value; "
                "n_complete / n_failed: its trials)"
            ),
            mo.md(f"```text\n{model1_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Trials per feature transformer**"),
            mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
        ]
    )


if __name__ == "__main__":
    app.run()
