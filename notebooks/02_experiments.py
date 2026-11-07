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
    independent cohorts model keeps the sharing flags of its runs in section
    6 (one set of hyperparameters, a transformer per cohort). The total times
    probability model is tuned by coordinate descent, as its runs in section 6: its
    total with the probabilities fixed at a plain logistic model, then its
    logistic probabilities with that best total fixed; its `cv_best` is the
    second study's value, the pair's, and its seconds are the two studies'.
    The expected caveat: the log loss scores each row's proportions only
    (the row divided by its sum), so under it the independent cohorts
    model's scale and the total times probability model's total get no
    signal (the total changes nothing in the log loss, so every trial of its
    study scores the same).

    Every study, here and in section 6, is stored in its own file under
    `studies/` (Optuna's journal file; git-ignored), and a click loads a
    stored study instead of tuning it, with the seconds its tuning took. The
    directory's name holds what shapes every study: the data's config hash,
    the split and its folds, the seed and the budget; the file's name holds
    what the study tunes (the model, or the total times probability model's
    coordinate-descent step), its run settings, its objective and its seed.
    So a changed setting is a new study, and a study run in both sections
    (each model's first run under the Poisson deviance) is tuned once. Delete
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
    # The objective EDA's objectives (section 5), the same objects as in METRICS.
    EDA_OBJECTIVES = (LOG_LOSS, POISSON_DEVIANCE, RMSE)
    # The selection metric: it picks each model's best run (the composite's
    # point score on validation) and is the forest plots' metric.
    if selection_metric == "rmse":
        SELECTION_METRIC = RMSE
    elif selection_metric == "poisson_deviance":
        SELECTION_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown selection_metric {selection_metric!r}")
    return EDA_OBJECTIVES, METRICS, POISSON_DEVIANCE, SELECTION_METRIC


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
    # descent's step 1, in the objective EDA and in section 6. The composite
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
def _(ProbabilityComponent, pd):
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
        rows = []
        for trial, components in zip(study.trials, components_by_trial, strict=True):
            feature_transformers, hyperparameters = {}, {}
            for component in components:
                candidates = feature_transformer_dict[component.name]
                feature_transformers[component.name] = next(
                    transformer_name
                    for transformer_name, transformer in candidates.items()
                    if transformer is component.feature_transformer
                )
                estimator_params = component.to_dict()["estimator_params"]
                if isinstance(component, ProbabilityComponent):
                    estimator_params["calibration_method"] = (
                        component.calibration_method
                    )
                hyperparameters[component.name] = estimator_params
            rows.append(
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
        return pd.DataFrame(rows)

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
    # settings of section 6's first run: its deviance study is that run's.
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
    # The total times probability model's EDA run, by coordinate descent as its
    # runs in section 6: per objective, step 1 tunes the total with the fixed
    # classifier, step 2 the probabilities with step 1's best total; every
    # study seeded seed (M8). Step 2's value is the pair's cv: its fixed total
    # is the pair's.
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
    METRICS,
    eda_model1_df,
    eda_model1_trials_df,
    eda_model2_df,
    eda_model2_trials_df,
    mo,
    pd,
):
    # Each model's best trial per objective: its study's minimum on the
    # objective (the first on ties, as Optuna's best trial; the total times
    # probability model's in its probability step, the pair the refit holds),
    # its cross-validated score on every metric, and the refit's composite on
    # validation.
    _metric_names = [metric.name for metric in METRICS]
    _rows = []
    for _model, _trials, _scores in (
        ("model1", eda_model1_trials_df, eda_model1_df),
        (
            "model2",
            eda_model2_trials_df[eda_model2_trials_df["step"] == "probability"],
            eda_model2_df,
        ),
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
                "best trial, the total times probability model's in its probability "
                "step; cv_: its cross-validated scores; val_: the refit's composite "
                "on validation)"
            ),
            mo.ui.table(eda_best_df.round(4)),
        ]
    )
    return (eda_best_df,)


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
    eda_model2_trials_df,
    eda_objective,
    go,
    make_subplots,
    mo,
):
    # Per model and objective, every metric per trial, a panel each on its own
    # scale, the trials in order on a shared axis (the total times probability
    # model's total step, then its probability step): the best trial a red star
    # in every panel. The colors are the dataviz reference palette's (the
    # trials series 1, the best series 8, the rule secondary ink, the chart
    # surface); the trials translucent, so overlapping ones show darker.
    _metric_names = [metric.name for metric in METRICS]
    _surface, _ink = "#fcfcfb", "#52514e"
    eda_figures = {}
    for _model, _trials in (
        ("model1", eda_model1_trials_df),
        ("model2", eda_model2_trials_df),
    ):
        for _objective, _study_trials in _trials.groupby("objective", sort=False):
            _study_trials = _study_trials.reset_index(drop=True)
            _best_id = eda_best_df.loc[
                (eda_best_df["model"] == _model)
                & (eda_best_df["objective"] == _objective),
                "best_id",
            ].iloc[0]
            if _model == "model1":
                _labels = "trial " + _study_trials["id"].astype(str)
                _is_best = _study_trials["id"] == _best_id
                _probability_start = None
            elif _model == "model2":
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
            eda_figures["model1", eda_objective.value],
            mo.md(
                "**The total times probability model: every metric per trial** (the "
                "total step's trials, then the probability step's; the star its best "
                "trial, in the probability step)"
            ),
            eda_figures["model2", eda_objective.value],
        ]
    )
    return


@app.cell
def _(METRICS, mo):
    # The recorded metric on which the EDA's objectives are compared.
    eda_recorded_metric = mo.ui.dropdown(
        options=[metric.name for metric in METRICS],
        value=METRICS[0].name,
        label="Recorded metric",
    )
    return (eda_recorded_metric,)


@app.cell
def _(
    METRICS,
    eda_best_df,
    eda_model1_trials_df,
    eda_model2_trials_df,
    eda_recorded_metric,
    go,
    mo,
):
    # Per model and recorded metric, the objectives' studies compared: a box per
    # objective with every trial a point and its best trial a star (the model it
    # picks); the total times probability model's two steps side by side. The
    # colors are the dataviz reference palette's first three categorical slots,
    # each objective with its own marker; the y-axis spans the bulk of the trials
    # (their 2nd to 98th percentile) and every star.
    _styles = {
        "log_loss": ("#2a78d6", "circle"),
        "poisson_deviance": ("#eb6834", "diamond"),
        "rmse": ("#1baf7a", "triangle-up"),
    }
    _surface, _ink = "#fcfcfb", "#52514e"
    eda_metric_figures = {}
    for _model, _trials in (
        ("model1", eda_model1_trials_df),
        ("model2", eda_model2_trials_df),
    ):
        if _model == "model1":
            _x_column, _boxmode = "objective", "overlay"
            _labels = "trial " + _trials["id"].astype(str)
            _best_candidates = _trials
        elif _model == "model2":
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
            eda_metric_figures["model1", eda_recorded_metric.value],
            mo.md(
                "**The total times probability model: the objectives compared on "
                "the chosen metric** (each step's trials; the star its best trial, "
                "in the probability step)"
            ),
            eda_metric_figures["model2", eda_recorded_metric.value],
        ]
    )
    return


@app.cell
def _(
    eda_model1_df,
    eda_model1_trials_df,
    eda_model2_df,
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
                "**The total times probability model** (the same; cv_best: step "
                "2's value, the pair's, in its own objective; seconds: the two "
                "studies')"
            ),
            mo.ui.table(eda_model2_df.round(4)),
            mo.md(
                "**Every trial: the independent cohorts model** (one row per trial "
                "of each objective's study, the whole model it scored: its state, "
                "seconds, cross-validated score on every metric, and each cohort's "
                "feature transformer and hyperparameters)"
            ),
            mo.ui.table(eda_model1_trials_df.round(4)),
            mo.md(
                "**Every trial: the total times probability model** (each "
                "objective's two studies, each trial by its step and number: step "
                "`total`, each drawn total with the fixed classifier; step "
                "`probability`, step 1's best total with each drawn classifier)"
            ),
            mo.ui.table(eda_model2_trials_df.round(4)),
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
    `n_trials = 100`, the first `n_startup_trials = 10` drawn at random.
    Each study is stored, and loaded by the next click (section 5).
    The runs go one after another, seeded
    `seed + 0`, `seed + 1`, … in order (each model's in loops over its run
    settings), and each trial fits its 5 folds in
    parallel. A trial whose fit raises (an NB2 fit that does not converge) is
    marked failed and the run goes on; the tables count them. Each run keeps
    its best model only, not its trials (section 5 studies them): its
    components, read back from the best trial's parameters, their feature
    transformers' names, its value, its trial counts and seconds. The
    independent cohorts model's best, and the total times probability
    model's pairs (below), are refit on the training part, kept, and scored
    on the training rows and on the validation rows (the test part waits for
    the final refit on training + validation) on every metric, per cohort
    and for the composite: a point score each. Each model
    has two tables: its runs (each kept model's feature transformers and
    hyperparameters per component, its cross-validated objective, its trials
    and seconds) and their scores (a row per dataset, cohort and metric). The six
    metrics: the Poisson deviance (the primary metric, the tuning's
    objective); the log loss (the proportions between cohorts only, so it
    never selects); MAE and RMSE per (building, cohort) cell; WMAPE (the
    absolute error as a share of the children); and the bias (mean
    predicted minus true count, signed, so 0 is best). A Gaussian loss can predict a negative count,
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
    cross-validated and scored, with no search; a pair's trials and seconds
    are its two runs'. Its replication of the
    counts into classifier
    rows is fixed at `weighted`, not searched: for the logistic classifier
    the two replications are the same fit, and for LightGBM they differed
    within noise while `per_child` is 2-3× slower
    (`docs/TOTAL_TIMES_PROBABILITY_MODEL.md` §3.1).
    The run
    cells sit behind buttons: opening the notebook fits nothing.

    Each model's best run is the one with the lowest point score of the
    composite on validation by `selection_metric` (RMSE by default, or the
    Poisson deviance). The point score, because the bootstrap's mean ranks
    the runs the same (the bootstrap measures the uncertainty, it does not
    rank better), and not the cross-validated score, since each run was
    tuned on it. The chosen run's validation score is optimistic (it won a
    choice); the test part gives the fair one. The two best runs are
    bootstrapped on validation: each score with its replicates' mean,
    median, SD and `1 - alpha` (95%) interval, `ci95_lower` to
    `ci95_upper`. The replicates resample whole neighborhoods, and both
    models are scored on the same replicates, so their intervals compare
    and their difference pairs by replicate; `n_resamples = 1000` was kept
    by measurement (the bounds' SD across seeds at most 4.6% of the
    interval's width). A forest plot shows the selection metric with its
    interval per cohort and for the composite.

    The two best runs are then inspected, behind their own button, with
    their stored models. The permutation importance of the primary metric on validation
    says what the shipped prediction depends on, and on training what the
    model fitted (a column important there only is overfit). Exact SHAP of
    `n_shap_rows` validation rows against `n_shap_background` training rows
    attributes the predicted count per row, column and cohort; mean |SHAP|
    per column is its global importance. The last table puts the two
    importances' rankings side by side.
    """)
    return


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
        # trials are not kept: section 5 studies them.
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
        # counts and seconds. The trials are not kept: section 5 studies them.
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
    return best_model1, best_model2, best_replicate_scores, best_runs_val_scores_df


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
def _(CI_BOUND_NAMES, SELECTION_METRIC, best_runs_val_scores_df, forest_figure, mo):
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
    return (best_forest_figure,)


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
