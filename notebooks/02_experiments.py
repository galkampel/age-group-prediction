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
    # The independent cohorts model's search: one set of hyperparameters for
    # all cohorts, and one feature transformer for all cohorts (D24.4); the
    # other arm is a click with share_feature_transformer = True.
    share_hyperparams = True
    share_feature_transformer = False
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
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
        share_feature_transformer,
        share_hyperparams,
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
    ## 5) Hyperparameter tuning

    Two models: the independent cohorts model (Model 1, one count model per
    cohort) and the total times probability model (Model 2, a count model of
    the total times a classifier's cohort probabilities). One run per model
    family and exposure (on or off, fixed for the run), each in its own cell:
    one Optuna study on the composite, so the model that is scored is the
    model that is tuned, by the cohort Poisson deviance of its
    cross-validation folds on the training part, the plain mean over folds.
    A trial draws, per cohort (per part for the total times probability
    model), the estimator's hyperparameters and the feature transformer,
    under the names `<component>__<parameter>` (the transformer by its name,
    resolved to the object); they make a component, a complete record of
    that part, and the composite is built from the components with
    constructor calls; the best run's components are read back from the
    study's `best_params` by the component type, the same records. Two
    settings decide what the independent cohorts model's cohorts share,
    each drawn once under `shared__`:
    `share_hyperparams` (one set of hyperparameters for all cohorts) and
    `share_feature_transformer` (one transformer for all cohorts). The runs
    below take `True` and `False`; the other arm, both `True`, is a click
    with the setting changed. LightGBM's two losses, Poisson and Gaussian, are
    separate families, and the exposure a run setting: a study that sampled
    the loss split 30 trials 10/20 and did worse than either loss alone, and
    the exposure changes the target (a count or a weighted rate).
    `n_trials = 400`, the first `n_startup_trials = 20` drawn at random; the
    tables of trials per choice show whether any was tried too rarely to be
    judged. The runs go one after another, seeded
    `seed + 0`, `seed + 1`, … in order, and each trial fits its 5 folds in
    parallel. A trial whose fit raises (an NB2 fit that does not converge) is
    marked failed and the run goes on; the tables count them. Each run keeps
    every trial and its best components, which alone are read back from the
    best trial's parameters, refit on the training part and scored on
    training and validation. A Gaussian loss can predict a negative count,
    which `CountModel` floors at `y_min` (eps). The total times probability
    model is tuned the same way, one run per total family, the total's
    exposure and a classifier family, 16 runs, under `total__…` and
    `probability__…`; the probability model ignores the exposure, and its
    calibration is searched. Its replication of the counts into classifier
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
    that resamples whole neighborhoods gives the interval of seven metrics,
    all paired on the same replicates: the cohort Poisson deviance (the
    primary metric, the tuning's objective); the cohort log loss (the
    proportions between cohorts only, so it never selects); D² Poisson (the
    share of the deviance explained, 1 is perfect); MAE and RMSE per
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
    from functools import partial

    import numpy as np
    import optuna
    from sklearn.metrics import (
        d2_tweedie_score,
        mean_absolute_error,
        mean_poisson_deviance,
        root_mean_squared_error,
    )

    from age_group_prediction.hyperparameter_tuning import (
        CountComponent,
        CrossValidationTuner,
        ProbabilityComponent,
        ProbabilitySearchSpace,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        fit_and_score,
    )
    from age_group_prediction.inspection import (
        BootstrapEvaluator,
        PermutationImportance,
        ShapExplainer,
    )
    from age_group_prediction.scoring import (
        Metric,
        cohort_log_loss,
        cohort_poisson_deviance,
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
        ProbabilitySearchSpace,
        ShapExplainer,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        cohort_log_loss,
        cohort_poisson_deviance,
        d2_tweedie_score,
        fit_and_score,
        mean_absolute_error,
        mean_poisson_deviance,
        np,
        partial,
        root_mean_squared_error,
    )


@app.cell
def _(
    Metric,
    cohort_log_loss,
    cohort_poisson_deviance,
    d2_tweedie_score,
    mean_absolute_error,
    mean_poisson_deviance,
    np,
    partial,
    root_mean_squared_error,
):
    # The study's objective and the tables' per-cohort score, with the direction
    # Optuna is given; a loss, so "minimize".
    COHORT_POISSON_DEVIANCE = Metric(
        "cohort_poisson_deviance", cohort_poisson_deviance, direction="minimize"
    )
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance, direction="minimize"
    )
    COHORT_LOG_LOSS = Metric("cohort_log_loss", cohort_log_loss, direction="minimize")

    def on_cells(function):
        """``function`` on the flattened (building, cohort) cells of both matrices."""
        return lambda y_true, y_pred: function(
            np.asarray(y_true, dtype=float).ravel(),
            np.asarray(y_pred, dtype=float).ravel(),
        )

    def wmape(y_true, y_pred):
        return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())

    def bias(y_true, y_pred):
        return float(np.mean(y_pred - y_true))

    # The bootstrap of the best runs, the primary metric first. Bias is signed,
    # so its direction is nominal: only the permutation importance reads one.
    INSPECTION_METRICS = (
        COHORT_POISSON_DEVIANCE,
        COHORT_LOG_LOSS,
        Metric(
            "d2_poisson",
            on_cells(partial(d2_tweedie_score, power=1)),
            direction="maximize",
        ),
        Metric("mae", on_cells(mean_absolute_error), direction="minimize"),
        Metric("rmse", on_cells(root_mean_squared_error), direction="minimize"),
        Metric("wmape", on_cells(wmape), direction="minimize"),
        Metric("bias", on_cells(bias), direction="minimize"),
    )
    return COHORT_POISSON_DEVIANCE, INSPECTION_METRICS, POISSON_DEVIANCE


@app.cell
def _(Y, np):
    COHORTS = list(Y.columns)
    # The smallest mean a CountModel predicts: a Gaussian loss can predict a
    # negative count, the Poisson deviance raises at 0 and stays finite at eps.
    PREDICTION_FLOOR = float(np.finfo(float).eps)
    return COHORTS, PREDICTION_FLOOR


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    CrossValidationTuner,
    cv,
    n_splits,
    n_startup_trials,
    n_trials,
):
    # The study's design, the same for every run: the cohort Poisson deviance
    # of the folds, the plain mean over them, the first n_startup_trials of
    # n_trials at random; each trial fits its folds in n_splits processes
    # (D13, D20, D26).
    tuner = CrossValidationTuner(
        cv=cv,
        metric=COHORT_POISSON_DEVIANCE,
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        n_jobs=n_splits,
    )
    return (tuner,)


@app.cell
def _(mo):
    run_model1 = mo.ui.run_button(label="Tune the independent cohorts model")
    mo.md(
        f"{run_model1} 8 runs, one cell each (a count family × exposure on or off); "
        "each trial fits its folds in parallel."
    )
    return (run_model1,)


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 1 of 8: LightGBM with the Poisson loss,
    # exposure on.
    _family, _use_exposure = "lgbm_poisson", True
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 0,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 2 of 8: LightGBM with the Poisson loss,
    # exposure off.
    _family, _use_exposure = "lgbm_poisson", False
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 1,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 3 of 8: LightGBM with the Gaussian loss,
    # exposure on.
    _family, _use_exposure = "lgbm_gaussian", True
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 2,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 4 of 8: LightGBM with the Gaussian loss,
    # exposure off.
    _family, _use_exposure = "lgbm_gaussian", False
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 3,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 5 of 8: the Poisson GLM, exposure on.
    _family, _use_exposure = "poisson_glm", True
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 4,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 6 of 8: the Poisson GLM, exposure off.
    _family, _use_exposure = "poisson_glm", False
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 5,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 7 of 8: the negative binomial GLM (NB2),
    # exposure on.
    _family, _use_exposure = "nb2", True
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 6,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
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
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    fit_and_score,
    mo,
    run_model1,
    seed,
    share_feature_transformer,
    share_hyperparams,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model, run 8 of 8: the negative binomial GLM (NB2),
    # exposure off.
    _family, _use_exposure = "nb2", False
    _study, _seconds = tuner.tune_independent_cohorts_model(
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv_groups,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=PREDICTION_FLOOR,
        random_state=seed,
        study_seed=seed + 7,
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
    _, _score_table = fit_and_score(
        build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
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
        "share_hyperparams": share_hyperparams,
        "share_feature_transformer": share_feature_transformer,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model1_nb2_no_exposure,)


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
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
        _settings = {
            "family": _run["family"],
            "use_exposure": _run["use_exposure"],
            "share_hyperparams": _run["share_hyperparams"],
            "share_feature_transformer": _run["share_feature_transformer"],
        }
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
    return MODEL1_RUNS, model1_arms_df, model1_df


@app.cell
def _(mo, model1_arms_df, model1_df):
    mo.vstack(
        [
            mo.md(
                "**The independent cohorts model: cohort Poisson deviance** (cv_best: the run's value; "
                "n_complete / n_failed: its trials)"
            ),
            mo.md(f"```text\n{model1_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Trials per feature transformer**"),
            mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(mo):
    run_model2 = mo.ui.run_button(label="Tune the total times probability model")
    mo.md(
        f"{run_model2} 16 runs, one cell each (a total family × exposure on or off "
        "× a probability family); each trial fits its folds in parallel."
    )
    return (run_model2,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 1 of 16: the LightGBM total with the
    # Poisson loss, exposure on;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = "lgbm_poisson", "logistic", True
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
        study_seed=seed + 0,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_poisson_logistic_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_poisson_logistic_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 2 of 16: the LightGBM total with the
    # Poisson loss, exposure on;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "lgbm_poisson", "lgbm", True
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
        study_seed=seed + 1,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_poisson_lgbm_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_poisson_lgbm_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 3 of 16: the LightGBM total with the
    # Poisson loss, exposure off;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = (
        "lgbm_poisson",
        "logistic",
        False,
    )
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
        study_seed=seed + 2,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_poisson_logistic_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_poisson_logistic_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 4 of 16: the LightGBM total with the
    # Poisson loss, exposure off;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "lgbm_poisson", "lgbm", False
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
        study_seed=seed + 3,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_poisson_lgbm_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_poisson_lgbm_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 5 of 16: the LightGBM total with the
    # Gaussian loss, exposure on;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = (
        "lgbm_gaussian",
        "logistic",
        True,
    )
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
        study_seed=seed + 4,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_gaussian_logistic_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_gaussian_logistic_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 6 of 16: the LightGBM total with the
    # Gaussian loss, exposure on;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "lgbm_gaussian", "lgbm", True
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
        study_seed=seed + 5,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_gaussian_lgbm_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_gaussian_lgbm_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 7 of 16: the LightGBM total with the
    # Gaussian loss, exposure off;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = (
        "lgbm_gaussian",
        "logistic",
        False,
    )
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
        study_seed=seed + 6,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_gaussian_logistic_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_gaussian_logistic_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 8 of 16: the LightGBM total with the
    # Gaussian loss, exposure off;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "lgbm_gaussian", "lgbm", False
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
        study_seed=seed + 7,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_lgbm_gaussian_lgbm_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_lgbm_gaussian_lgbm_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 9 of 16: the Poisson GLM total, exposure
    # on;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = "poisson_glm", "logistic", True
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
        study_seed=seed + 8,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_poisson_glm_logistic_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_poisson_glm_logistic_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 10 of 16: the Poisson GLM total, exposure
    # on;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "poisson_glm", "lgbm", True
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
        study_seed=seed + 9,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_poisson_glm_lgbm_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_poisson_glm_lgbm_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 11 of 16: the Poisson GLM total, exposure
    # off;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = "poisson_glm", "logistic", False
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
        study_seed=seed + 10,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_poisson_glm_logistic_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_poisson_glm_logistic_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 12 of 16: the Poisson GLM total, exposure
    # off;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "poisson_glm", "lgbm", False
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
        study_seed=seed + 11,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_poisson_glm_lgbm_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_poisson_glm_lgbm_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 13 of 16: the negative binomial GLM (NB2)
    # total, exposure on;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = "nb2", "logistic", True
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
        study_seed=seed + 12,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_nb2_logistic_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_nb2_logistic_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 14 of 16: the negative binomial GLM (NB2)
    # total, exposure on;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "nb2", "lgbm", True
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
        study_seed=seed + 13,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_nb2_lgbm_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_nb2_lgbm_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 15 of 16: the negative binomial GLM (NB2)
    # total, exposure off;
    # logistic probabilities.
    _total_family, _probability_family, _use_exposure = "nb2", "logistic", False
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
        study_seed=seed + 14,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_nb2_logistic_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_nb2_logistic_no_exposure,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    POISSON_DEVIANCE,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    fit_and_score,
    mo,
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
    # The total times probability model, run 16 of 16: the negative binomial GLM (NB2)
    # total, exposure off;
    # LightGBM probabilities.
    _total_family, _probability_family, _use_exposure = "nb2", "lgbm", False
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
        study_seed=seed + 15,
    )
    # The best components, read from the best trial's parameters by their
    # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    )
    _best_components = (_total_component, _probability_component)
    _, _score_table = fit_and_score(
        build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ),
        train_age_group,
        val_age_group,
        cohort_metric=POISSON_DEVIANCE,
        metric=COHORT_POISSON_DEVIANCE,
    )
    # Every trial and the best (D14); the study object is not kept.
    model2_nb2_lgbm_no_exposure = {
        "total_family": _total_family,
        "probability_family": _probability_family,
        "use_exposure": _use_exposure,
        "trials": _study.trials_dataframe(),
        "components": [component.to_dict() for component in _best_components],
        "cv_best": _study.best_value,
        "best_params": _study.best_params,
        "score_table": _score_table,
        "seconds": _seconds,
        "seed": seed,
        "y_min": PREDICTION_FLOOR,
    }
    return (model2_nb2_lgbm_no_exposure,)


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilitySearchSpace,
    model2_lgbm_gaussian_lgbm_exposure,
    model2_lgbm_gaussian_lgbm_no_exposure,
    model2_lgbm_gaussian_logistic_exposure,
    model2_lgbm_gaussian_logistic_no_exposure,
    model2_lgbm_poisson_lgbm_exposure,
    model2_lgbm_poisson_lgbm_no_exposure,
    model2_lgbm_poisson_logistic_exposure,
    model2_lgbm_poisson_logistic_no_exposure,
    model2_nb2_lgbm_exposure,
    model2_nb2_lgbm_no_exposure,
    model2_nb2_logistic_exposure,
    model2_nb2_logistic_no_exposure,
    model2_poisson_glm_lgbm_exposure,
    model2_poisson_glm_lgbm_no_exposure,
    model2_poisson_glm_logistic_exposure,
    model2_poisson_glm_logistic_no_exposure,
    pd,
):
    MODEL2_RUNS = [
        model2_lgbm_poisson_logistic_exposure,
        model2_lgbm_poisson_lgbm_exposure,
        model2_lgbm_poisson_logistic_no_exposure,
        model2_lgbm_poisson_lgbm_no_exposure,
        model2_lgbm_gaussian_logistic_exposure,
        model2_lgbm_gaussian_lgbm_exposure,
        model2_lgbm_gaussian_logistic_no_exposure,
        model2_lgbm_gaussian_lgbm_no_exposure,
        model2_poisson_glm_logistic_exposure,
        model2_poisson_glm_lgbm_exposure,
        model2_poisson_glm_logistic_no_exposure,
        model2_poisson_glm_lgbm_no_exposure,
        model2_nb2_logistic_exposure,
        model2_nb2_lgbm_exposure,
        model2_nb2_logistic_no_exposure,
        model2_nb2_lgbm_no_exposure,
    ]
    _score_rows, _arm_rows = [], []
    for _run in MODEL2_RUNS:
        # A family pair reads as one family, so both models' tables have the same
        # settings columns; the record keeps the two families (D16).
        _settings = {
            "family": f"{_run['total_family']}+{_run['probability_family']}",
            "use_exposure": _run["use_exposure"],
        }
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
        # The completed trials per categorical choice, every candidate listed, so
        # one never tried shows 0 (D15).
        _candidates = {
            "params_total__feature_transformer": list(
                COUNT_FEATURE_TRANSFORMERS[_run["total_family"]]
            ),
            "params_probability__feature_transformer": list(
                PROBABILITY_FEATURE_TRANSFORMERS[_run["probability_family"]]
            ),
            "params_probability__calibration_method": (
                ProbabilitySearchSpace.CALIBRATION_METHODS
            ),
        }
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
    model2_df, model2_arms_df = pd.DataFrame(_score_rows), pd.DataFrame(_arm_rows)
    return MODEL2_RUNS, model2_arms_df, model2_df


@app.cell
def _(mo, model2_arms_df, model2_df):
    mo.vstack(
        [
            mo.md(
                "**The total times probability model: cohort Poisson deviance** (family: the total's + the "
                "probabilities'; cv_best: the run's value; n_complete / n_failed: "
                "its trials)"
            ),
            mo.md(f"```text\n{model2_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Trials per choice**"),
            mo.md(f"```text\n{model2_arms_df.to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(mo, model1_df, model2_df, pd):
    # Each model's best run on validation, from the composite rows. A fresh
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
        ["model", "family", "use_exposure", "cv_best", "train", "val"],
    ]
    mo.vstack(
        [
            mo.md("**Best run per model** (validation cohort Poisson deviance)"),
            mo.md(f"```text\n{best_df.round(4).to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: refit on the training part, the "
        "bootstrap of seven metrics, the permutation importance and SHAP (about two "
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
    best_run1 = min(MODEL1_RUNS, key=lambda run: run["score_table"][-1]["val"])
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
    best_run2 = min(MODEL2_RUNS, key=lambda run: run["score_table"][-1]["val"])
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
    INSPECTION_METRICS,
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
    # The seven metrics' intervals on the training rows and on the validation
    # rows, resampling whole neighborhoods; every metric scores the same replicates.
    _evaluator = BootstrapEvaluator(n_resamples, random_state=seed)
    model1_intervals = pd.concat(
        {
            "train": _evaluator.evaluate(
                best_model1,
                train_age_group.X,
                train_age_group.Y,
                train_age_group.groups,
                exposure=train_age_group.exposure,
                metrics=INSPECTION_METRICS,
            ),
            "val": _evaluator.evaluate(
                best_model1,
                val_age_group.X,
                val_age_group.Y,
                val_age_group.groups,
                exposure=val_age_group.exposure,
                metrics=INSPECTION_METRICS,
            ),
        },
        names=["split", "metric"],
    )
    return (model1_intervals,)


@app.cell
def _(
    BootstrapEvaluator,
    INSPECTION_METRICS,
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
                best_model2,
                train_age_group.X,
                train_age_group.Y,
                train_age_group.groups,
                exposure=train_age_group.exposure,
                metrics=INSPECTION_METRICS,
            ),
            "val": _evaluator.evaluate(
                best_model2,
                val_age_group.X,
                val_age_group.Y,
                val_age_group.groups,
                exposure=val_age_group.exposure,
                metrics=INSPECTION_METRICS,
            ),
        },
        names=["split", "metric"],
    )
    return (model2_intervals,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
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
    _importance = PermutationImportance(
        COHORT_POISSON_DEVIANCE, n_repeats, random_state=seed
    )
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
    COHORT_POISSON_DEVIANCE,
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
    _importance = PermutationImportance(
        COHORT_POISSON_DEVIANCE, n_repeats, random_state=seed
    )
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
    # Each metric's score and interval per model and split; the caption names
    # the two best runs.
    intervals_df = pd.concat(
        {"model1": model1_intervals, "model2": model2_intervals}, names=["model"]
    ).reset_index()
    mo.vstack(
        [
            mo.md(
                "**The best runs' metrics** (score: the split's rows; mean, lower, "
                "upper: the bootstrap over neighborhoods). model1, the "
                f"independent cohorts model: `{best_run1['family']}`, exposure "
                f"{best_run1['use_exposure']}; model2, the total times probability "
                f"model: `{best_run2['total_family']}+{best_run2['probability_family']}`, "
                f"exposure {best_run2['use_exposure']}."
            ),
            mo.md(f"```text\n{intervals_df.round(4).to_string(index=False)}\n```"),
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
