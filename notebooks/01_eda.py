import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building-Level Student Population EDA

    This notebook implements six EDA steps for the compact simulator:

    1. Set up the local package and load the typed simulation configuration.
    2. Generate the expanded EDA population and audit structural, accounting,
       missingness, and leakage constraints.
    3. Characterize target and feature distributions, cohort shares,
       overdispersion, pairwise associations, and deterministic collinearity.
    4. Inspect marginal nonlinear patterns with raw support and binned summaries.
    5. Explore conditional nonlinearities with Poisson HGB partial-dependence and
       ICE views.
    6. Compare supported empirical interaction trajectories and room composition,
       then consolidate the observed evidence in the final findings register.

    The accepted findings from this notebook feed the modeling, selection,
    guarded held-out evaluation, and MLflow workflow in
    `notebooks/02_model_fitting.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Setup And Imports

    Import the simulator API and common analysis libraries.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pandas.api.types import is_integer_dtype

    from student_simulator import (
        StudentPopulationSimulator,
        load_simulation_config,
    )

    sns.set_theme(style="white")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    return (
        StudentPopulationSimulator,
        is_integer_dtype,
        load_simulation_config,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Load Configuration

    Load the canonical config so your notebook run matches the documented stage assumptions.
    """)
    return


@app.cell
def _(load_simulation_config, mo):
    project_root = mo.notebook_dir().parent
    config_path = project_root / "configs" / "simulation.toml"
    base_eda_config = load_simulation_config(config_path)
    # Override the number of neighborhoods and buildings per neighborhood for EDA purposes
    eda_config = base_eda_config.model_copy(
        update={
            "simulation": base_eda_config.simulation.model_copy(
                update={"n_neighborhoods": 150}
            ),
            "building": base_eda_config.building.model_copy(
                update={"buildings_per_neighborhood_rate": 9.0}
            ),
        }
    )
    mo.md(
        f"""
        **Reference configuration**

        - Path: `{config_path.relative_to(project_root)}`
        - Seed: `{eda_config.simulation.seed}`
        - Neighborhoods: `{eda_config.simulation.n_neighborhoods}`
        - Building rate: `{eda_config.building.buildings_per_neighborhood_rate}`
        """
    )
    return config_path, eda_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Generate The EDA Population

    Create a dedicated EDA run with the agreed settings:
    - `n_neighborhoods = 150`
    - `buildings_per_neighborhood_rate = 9.0`

    The resulting building-level table is the sole analysis input for Steps 1-3.
    """)
    return


@app.cell
def _(StudentPopulationSimulator, config_path, eda_config, mo):
    eda_simulator = StudentPopulationSimulator(eda_config)
    eda_final_df = eda_simulator.run()
    eda_result = eda_simulator.last_result
    assert eda_result is not None
    buildings_per_neighborhood = (
        eda_final_df["neighborhood_id"]
        .value_counts()
        .sort_index()
    )
    building_summary = buildings_per_neighborhood.describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).to_frame(name="buildings_per_neighborhood").round(2)
    mo.vstack(
        [
            mo.md(
                f"""
                **EDA run ready:** `{len(eda_final_df):,}` buildings across
                `{eda_final_df["neighborhood_id"].nunique()}` neighborhoods
                using `{config_path.name}`.
                """
            ),
            building_summary,
            eda_final_df.head(8).round(3),
        ]
    )
    return (eda_final_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Step 1: Structural And Quality Audit

    Check EDA-critical accounting identities, type/range sanity, and missingness. Broader construction invariants are validated in tests.
    """)
    return


@app.cell
def _():
    id_columns = ["building_id", "neighborhood_id"]
    numeric_feature_columns = [
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
    ]
    categorical_feature_columns = ["school_status"]
    cohort_target_columns = ["n_kindergarten", "n_elementary", "n_highschool"]
    target_columns = [*cohort_target_columns, "n_children_total"]
    return (
        categorical_feature_columns,
        cohort_target_columns,
        id_columns,
        numeric_feature_columns,
        target_columns,
    )


@app.cell
def _(
    cohort_target_columns,
    eda_final_df,
    is_integer_dtype,
    mo,
    pd,
    target_columns,
):
    def check_integer_dtype(df: pd.DataFrame, columns: list[str]) -> bool:
        """Check if specified columns have integer dtype."""
        return all(is_integer_dtype(df[column]) for column in columns)

    def check_sums(df: pd.DataFrame, columns: list[str], total_column: str) -> bool:
        """Check if the sum of specified columns equals the total column."""
        return (df[columns].sum(axis=1) == df[total_column]).all()

    def check_non_negative(df: pd.DataFrame, columns: list[str]) -> bool:
        """Check if specified columns have non-negative values."""
        return (df[columns] >= 0).all().all()

    def classify_dtype(series: pd.Series) -> str:
        """Classify dtypes into readable analysis buckets."""
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"
        if pd.api.types.is_string_dtype(dtype):
            return "string"
        if pd.api.types.is_object_dtype(dtype):
            non_null_values = series.dropna()
            if not non_null_values.empty and non_null_values.map(type).eq(str).all():
                return "object(str values)"
            return "object(mixed values)"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        return "other"

    room_columns = ["3_rooms", "4_rooms", "5_rooms", "6_rooms"]

    # Perform quality checks on the EDA DataFrame
    quality_checks = {
        "targets_integer_dtype": check_integer_dtype(df=eda_final_df, columns=target_columns),
        "targets_non_negative": check_non_negative(df=eda_final_df, columns=target_columns),
        "rooms_sum_equals_n_apartments": (
            check_sums(eda_final_df, columns=room_columns, total_column="n_apartments")
        ),
        "cohorts_sum_equals_total": (
            check_sums(eda_final_df, columns=cohort_target_columns, total_column="n_children_total")
        ),
    }
    quality_summary_df = pd.DataFrame(
        {
            "check": list(quality_checks.keys()),
            "passed": list(quality_checks.values()),
        }
    )

    if not quality_summary_df["passed"].all():
        failed_checks = quality_summary_df.loc[
            ~quality_summary_df["passed"], "check"
        ].tolist()
        raise AssertionError(f"Structural checks failed: {failed_checks}")

    # Build a dtype table with semantic buckets for non-numeric columns.
    dtype_summary_df = pd.DataFrame(
        {
            "dtype": eda_final_df.dtypes.astype(str),
            "logical_type": [classify_dtype(eda_final_df[column]) for column in eda_final_df.columns],
        },
        index=eda_final_df.columns,
    )
    dtype_summary_df.index.name = "column"

    # Compute missingness summary.
    missingness_df = eda_final_df.isna().sum().rename("missing_count").to_frame()
    missingness_df["missing_rate"] = missingness_df["missing_count"] / len(eda_final_df)
    missingness_nonzero_df = missingness_df.loc[missingness_df["missing_count"] > 0].sort_values(
        by=["missing_count", "missing_rate"],
        ascending=False,
    )

    if missingness_nonzero_df.empty:
        missingness_caption = "No missing values detected."
        missingness_display_df = missingness_df
    else:
        missingness_caption = (
            f"Columns with missing values: `{len(missingness_nonzero_df)}` "
            f"of `{missingness_df.shape[0]}`."
        )
        missingness_display_df = missingness_nonzero_df

    mo.vstack(
        [
            mo.md(
                f"**Shape:** `{eda_final_df.shape}`. "
                f"**Missing values:** "
                f"`{missingness_df['missing_count'].sum()}`."
            ),
            dtype_summary_df,
            quality_summary_df,
            mo.md(missingness_caption),
            *([missingness_display_df] if not missingness_nonzero_df.empty else []),
        ]
    )
    return quality_summary_df, room_columns


@app.cell
def _(
    categorical_feature_columns,
    eda_final_df,
    id_columns,
    numeric_feature_columns,
    pd,
):
    # Compute unique counts for each feature.
    feature_unique_counts_df = pd.DataFrame(
        {
            "feature": [
                *id_columns,
                *numeric_feature_columns,
                *categorical_feature_columns,
            ],
            "n_unique": [
                eda_final_df[_column].nunique(dropna=False)
                for _column in [
                    *id_columns,
                    *numeric_feature_columns,
                    *categorical_feature_columns,
                ]
            ],
        }
    )

    feature_unique_counts_df
    return


@app.cell
def _(eda_final_df, pd, room_columns):
    def create_share_df(df: pd.DataFrame, share_columns: list[str], total_column: str, mask: pd.Series | None = None) -> pd.DataFrame:
        """Create a DataFrame with shares of specified columns relative to a total column."""
        if mask is not None:
            df = df.loc[mask]
        return df[share_columns].div(df[total_column], axis=0)

    room_share_df: pd.DataFrame= create_share_df(
        df=eda_final_df,
        share_columns=room_columns,
        total_column="n_apartments",
        mask=eda_final_df["n_apartments"] > 0,
    )
    # Compute descriptive statistics for the number of apartments and room shares.
    (
        pd.concat([eda_final_df[["n_apartments"]], room_share_df], axis=1)
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
        .drop(index=["count"])
        .T
    )
    return (create_share_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Step 2: Target And Feature Distributions

    Summarize target behavior, overdispersion indicators, cohort-share behavior, feature distributions, and key stratified comparisons.
    """)
    return


@app.cell
def _(
    cohort_target_columns,
    create_share_df,
    eda_final_df,
    mo,
    np,
    pd,
    target_columns,
):
    def summarize_count_target(df: pd.DataFrame, target_column: str) -> dict[str, int | float | str]:
        series: pd.Series = df[target_column].astype(float)
        mean = float(series.mean())
        variance = float(series.var(ddof=1))
        data = {
            "target": target_column,
            "mean": mean,
            "std": float(series.std(ddof=1)),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.5)),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "zero_share": float((series == 0).mean()),
            "variance_to_mean": variance / mean if mean > 0 else np.nan,
        }
        return data


    target_distribution_summary_df = (
        pd.DataFrame(
            [summarize_count_target(eda_final_df, target_column=target_col) for target_col in target_columns]
        )
        .round(3)
    )
    positive_total_mask = eda_final_df["n_children_total"] > 0
    cohort_share_df = create_share_df(
        df=eda_final_df,
        share_columns=cohort_target_columns,
        total_column="n_children_total",
        mask=positive_total_mask,
    )
    cohort_share_summary_df = (
        cohort_share_df.describe(percentiles=[0.1, 0.5, 0.9])
        .drop(index=["count"])
        .T.round(3)
    )

    mo.vstack(
        [
            target_distribution_summary_df,
            mo.md(
                f"**Positive-total buildings:** "
                f"`{int(positive_total_mask.sum())}`; "
                f"**zero-total buildings:** "
                f"`{int((~positive_total_mask).sum())}`."
            ),
            cohort_share_summary_df,
        ]
    )
    return (target_distribution_summary_df,)


@app.cell
def _(eda_final_df, plt, sns, target_columns):
    target_figure, target_axes = plt.subplots(
        len(target_columns),
        1,
        figsize=(6.5, 3.2 * len(target_columns)),
        layout="tight",
    )
    for row_index, target_column in enumerate(target_columns):
        sns.histplot(
            data=eda_final_df,
            x=target_column,
            discrete=True,
            stat="probability",
            ax=target_axes[row_index] if len(target_columns) > 1 else target_axes,
        )
        (target_axes[row_index] if len(target_columns) > 1 else target_axes).set_title(
            f"{target_column}: probability histogram"
        )
        (target_axes[row_index] if len(target_columns) > 1 else target_axes).set_xlabel(target_column)
        (target_axes[row_index] if len(target_columns) > 1 else target_axes).set_ylabel("probability")


    plt.show()
    return


@app.cell
def _(eda_final_df, is_integer_dtype, numeric_feature_columns, plt, sns):
    feature_figure, feature_axes = plt.subplots(
        2, 4, figsize=(15, 8), layout="constrained"
    )
    for axis, feature_column in zip(
        feature_axes.flat, numeric_feature_columns, strict=False
    ):
        sns.histplot(
            data=eda_final_df,
            x=feature_column,
            kde=not is_integer_dtype(eda_final_df[feature_column]),
            ax=axis
        )
        # axis.set_title(feature_column)

    school_figure, school_axis = plt.subplots(figsize=(7, 4))
    sns.countplot(
        data=eda_final_df,
        x="school_status",
        order=sorted(eda_final_df["school_status"].unique()),
        ax=school_axis,
    )
    school_axis.set_title("school_status frequency")
    school_axis.set_xlabel("school_status")
    school_axis.set_ylabel("count")


    plt.show()
    return


@app.cell
def _(eda_final_df, mo, pd, target_columns):
    def bin_feature_by_quantiles(df: pd.DataFrame, feature_column: str, n_bins: int = 4, method: str = "first", labels: list[str] | None = None) -> pd.Series:
        """Bin a numeric feature into quantiles."""
        if labels is None:
            labels = [f"Q{i+1}" for i in range(n_bins)]
        return pd.qcut(
            df[feature_column].rank(method=method),
            q=n_bins,
            labels=labels,
        )

    def calc_average_feature_effect_on_targets(
        df: pd.DataFrame,
        feature_col: str,
        target_columns: list[str],
        feature_order: list[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate the mean of target columns grouped by a feature column."""
        by_feature_df = df.groupby(
            feature_col, observed=False
        )[target_columns].mean().round(3)
        if feature_order is not None:
            by_feature_df = by_feature_df.reindex(feature_order).T
        else:
            by_feature_df = by_feature_df.T
        return by_feature_df

    def calc_average_feature_quantiles_on_targets(df: pd.DataFrame, qunatile_series: pd.Series, target_columns: list[str], quantile_feature_name: str) -> pd.DataFrame:
        """Calculate the mean of target columns grouped by quantile bins."""
        by_quantile_df = df.assign(
            **{quantile_feature_name: qunatile_series}
        ).groupby(quantile_feature_name, observed=False)[target_columns].mean().round(3)
        return by_quantile_df
    # Compute quantile bins for apartment counts and room composition.
    apartment_count_bin = bin_feature_by_quantiles(
        eda_final_df,
        "n_apartments",
        n_bins=4,
        method="first",
        labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"],
    )
    large_room_share = (
        ((eda_final_df["5_rooms"] + eda_final_df["6_rooms"]) / eda_final_df["n_apartments"])
        .to_frame(name="large_room_share")
    )
    room_mix_bin = bin_feature_by_quantiles(
        large_room_share,
        "large_room_share",
        n_bins=4,
        method="first",
        labels=["Q1_low_large_room_share", "Q2", "Q3", "Q4_high_large_room_share"],
    )
    # Compute target summaries by school status
    school_status_order = ["none", "planned", "existing"]
    by_school_status_df = calc_average_feature_effect_on_targets(
        eda_final_df,
        feature_col="school_status",
        target_columns=target_columns,
        feature_order=school_status_order,
    )
    # Compute target summaries by apartment-count
    by_apartment_bin_df = calc_average_feature_quantiles_on_targets(
        eda_final_df,
        qunatile_series=apartment_count_bin,
        target_columns=target_columns,
        quantile_feature_name="apartment_count_bin",
    )

    by_room_mix_bin_df = calc_average_feature_quantiles_on_targets(
        eda_final_df,
        qunatile_series=room_mix_bin,
        target_columns=target_columns,
        quantile_feature_name="room_mix_bin",
    )

    by_neighborhood_df = eda_final_df.groupby(
        "neighborhood_id", observed=False
    )[target_columns].mean()

    mo.vstack(
        [
            mo.md("### Target summary by school status"),
            by_school_status_df,
            mo.md("### Target summary by apartment-count quantile"),
            by_apartment_bin_df.T,
            mo.md("### Target summary by room-composition quantile"),
            by_room_mix_bin_df.T,
            mo.md("### Neighborhood-level mean target summary"),
            by_neighborhood_df.describe(percentiles=[0.1, 0.5, 0.9]).round(3).T.drop(columns=["count"]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Step 3: Correlation And Deterministic Dependency

    Compare Pearson and Spearman associations. This is not a general collinearity
    diagnostic. The structural audit separately establishes one exact dependency:
    the 3-6 room counts sum to `n_apartments` for every building. The modeling
    implication and recommended composition-aware treatment are summarized in
    Findings and Next Steps.
    """)
    return


@app.cell
def _(eda_final_df, numeric_feature_columns, pd, plt, sns, target_columns):
    def create_a_heatmap(
        df: pd.DataFrame,
        ax: plt.Axes,
        annot: bool = True,
        fmt: str = ".3f",
        cmap: str = "coolwarm",
        center: float = 0.0,
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> plt.Axes:
        """Create a heatmap for the given DataFrame."""
        sns.heatmap(
            df,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        return ax


    correlation_columns = [*numeric_feature_columns, *target_columns]
    pearson_corr_df = eda_final_df[correlation_columns].corr(method="pearson")
    spearman_corr_df = eda_final_df[correlation_columns].corr(method="spearman")

    corr_figure, corr_axes = plt.subplots(
        2, 1, figsize=(10, 16), layout="constrained"
    )

    corr_axes[0] = create_a_heatmap(
        pearson_corr_df,
        ax=corr_axes[0],
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
    )
    corr_axes[0].set_title("Pearson correlation matrix")
    corr_axes[1] = create_a_heatmap(
        spearman_corr_df,
        ax=corr_axes[1],
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
    )
    corr_axes[1].set_title("Spearman correlation matrix")

    feature_target_corr_df = pd.concat(
        {
            "pearson": pearson_corr_df.loc[numeric_feature_columns, target_columns],
            "spearman": spearman_corr_df.loc[numeric_feature_columns, target_columns],
        },
        axis=1,
    )
    feature_target_corr_display = feature_target_corr_df.T.copy()
    feature_target_corr_display.index = [
        f"{target_column} ({method})"
        for method, target_column in feature_target_corr_display.index
    ]
    feature_target_corr_display.index.name = None
    feature_target_corr_figure, feature_target_corr_axis = plt.subplots(
        figsize=(12, 7), layout="constrained"
    )
    feature_target_corr_axis = create_a_heatmap(
        feature_target_corr_display,
        ax=feature_target_corr_axis,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
    )
    feature_target_corr_axis.set_title("Feature-target correlations")
    feature_target_corr_axis.set_xlabel("feature")
    feature_target_corr_axis.set_ylabel("target (correlation method)")
    feature_target_corr_axis.tick_params(axis="x", labelrotation=90)


    plt.show()
    return (feature_target_corr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Step 4: Visual Marginal Nonlinearities

    These plots describe marginal building-level associations; they are not
    conditional effects or causal estimates. Raw hexbin panels retain the observed
    support. Quantile-bin panels show the mean with a 95% normal-approximation
    confidence interval, the median, the target interquartile range, the bin count,
    and the observed feature range. A shape-preserving spline is drawn only through
    bins meeting the displayed minimum-support rule.

    All outcomes in this section are counts. Cohort shares remain defined only in
    the earlier `n_children_total > 0` analysis and are not recomputed here.
    """)
    return


@app.cell
def _(eda_final_df, mo, plt, target_columns):
    marginal_support_threshold = 20

    daycare_target_mean_df = (
        eda_final_df.groupby("n_daycares_500m")[target_columns].mean()
    )
    daycare_building_counts = (
        eda_final_df["n_daycares_500m"].value_counts().sort_index()
    )
    daycare_sparse_mask = daycare_building_counts < marginal_support_threshold

    daycare_scatter_figure, daycare_scatter_axis = plt.subplots(
        figsize=(9, 5.5), layout="constrained"
    )
    for target_col in target_columns:
        means = daycare_target_mean_df[target_col]
        daycare_scatter_axis.plot(
            means.index,
            means,
            linewidth=1,
            alpha=0.35,
            zorder=1,
        )
        # Marker area tracks how many buildings back each mean.
        daycare_scatter_axis.scatter(
            means.index[~daycare_sparse_mask],
            means[~daycare_sparse_mask],
            s=20 + daycare_building_counts[~daycare_sparse_mask] * 0.35,
            label=target_col,
            zorder=2,
        )
        daycare_scatter_axis.scatter(
            means.index[daycare_sparse_mask],
            means[daycare_sparse_mask],
            s=60,
            facecolors="none",
            edgecolors="#b23a2b",
            linewidths=1.4,
            zorder=3,
        )

    for daycare_value, building_count in daycare_building_counts.items():
        daycare_scatter_axis.annotate(
            f"n={building_count}",
            (daycare_value, daycare_target_mean_df.loc[daycare_value].max()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="0.35",
        )

    daycare_scatter_axis.set(
        title="Mean children per building by n_daycares_500m",
        xlabel="n_daycares_500m",
        ylabel="mean children per building",
        xticks=daycare_target_mean_df.index,
    )
    daycare_scatter_axis.legend(title="target", fontsize=8, title_fontsize=9)

    mo.vstack(
        [
            mo.md(
                "Marker size tracks the number of buildings behind each mean. "
                f"Hollow red markers flag daycare values with fewer than "
                f"{marginal_support_threshold} "
                "buildings, where the mean is unstable."
            ),
            daycare_scatter_figure,
            daycare_target_mean_df.round(3).assign(
                n_buildings=daycare_building_counts
            ),
        ]
    )
    return (marginal_support_threshold,)


@app.cell
def _(np, pd, plt):
    def summarize_quantile_relationship(
        df: pd.DataFrame,
        feature_column: str,
        target_column: str,
        min_bin_count: int,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Summarize a count target over observed-value quantile bins."""
        analysis_df = df[[feature_column, target_column]].dropna().copy()
        available_bins = min(n_bins, int(analysis_df[feature_column].nunique()))
        analysis_df["feature_bin"] = pd.qcut(
            analysis_df[feature_column],
            q=available_bins,
            duplicates="drop",
        )
        summary_df = (
            analysis_df.groupby("feature_bin", observed=True)
            .agg(
                feature_min=(feature_column, "min"),
                feature_max=(feature_column, "max"),
                feature_center=(feature_column, "median"),
                target_mean=(target_column, "mean"),
                target_median=(target_column, "median"),
                target_std=(target_column, "std"),
                target_q25=(target_column, lambda values: values.quantile(0.25)),
                target_q75=(target_column, lambda values: values.quantile(0.75)),
                bin_count=(target_column, "size"),
            )
            .reset_index(drop=True)
        )
        summary_df["mean_se"] = summary_df["target_std"] / np.sqrt(
            summary_df["bin_count"]
        )
        summary_df["mean_ci_low"] = (
            summary_df["target_mean"] - 1.96 * summary_df["mean_se"]
        )
        summary_df["mean_ci_high"] = (
            summary_df["target_mean"] + 1.96 * summary_df["mean_se"]
        )
        summary_df["supported"] = summary_df["bin_count"] >= min_bin_count
        summary_df["feature_range"] = summary_df.apply(
            lambda row: f"{row['feature_min']:.2f}-{row['feature_max']:.2f}",
            axis=1,
        )
        summary_df.insert(0, "target", target_column)
        summary_df.insert(0, "feature", feature_column)
        summary_df.insert(1, "feature_range", summary_df.pop("feature_range"))
        return summary_df


    def create_marginal_nonlinearity_figure(
        df: pd.DataFrame,
        feature_column: str,
        targets: list[str],
        min_bin_count: int,
        n_bins: int = 10,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """Plot raw support and quantile-binned count summaries."""
        from scipy.interpolate import PchipInterpolator

        figure, figure_axes = plt.subplots(
            len(targets),
            2,
            figsize=(14, 3.6 * len(targets)),
            layout="constrained",
            squeeze=False,
        )
        summaries: list[pd.DataFrame] = []
        for row_index, target_column in enumerate(targets):
            summary_df = summarize_quantile_relationship(
                df,
                feature_column,
                target_column,
                n_bins=n_bins,
                min_bin_count=min_bin_count,
            )
            summaries.append(summary_df)
            raw_axis, binned_axis = figure_axes[row_index]
            raw_axis.hexbin(
                df[feature_column],
                df[target_column],
                gridsize=28,
                mincnt=1,
                cmap="Blues",
                linewidths=0,
            )
            raw_axis.set(
                title=f"{target_column}: raw building support",
                xlabel=feature_column,
                ylabel="count target",
            )

            positions = np.arange(len(summary_df))
            binned_axis.fill_between(
                positions,
                summary_df["target_q25"],
                summary_df["target_q75"],
                color="0.85",
                label="target IQR",
            )
            binned_axis.errorbar(
                positions,
                summary_df["target_mean"],
                yerr=[
                    summary_df["target_mean"] - summary_df["mean_ci_low"],
                    summary_df["mean_ci_high"] - summary_df["target_mean"],
                ],
                color="#1f5a85",
                marker="o",
                capsize=3,
                label="mean and 95% CI",
            )
            binned_axis.plot(
                positions,
                summary_df["target_median"],
                color="#a33b20",
                marker="s",
                linestyle="--",
                label="median",
            )

            supported_df = summary_df.loc[summary_df["supported"]].drop_duplicates(
                subset="feature_center"
            )
            if len(supported_df) >= 4:
                smooth_x = np.linspace(
                    supported_df["feature_center"].min(),
                    supported_df["feature_center"].max(),
                    100,
                )
                smooth_y = PchipInterpolator(
                    supported_df["feature_center"],
                    supported_df["target_mean"],
                )(smooth_x)
                raw_axis.plot(
                    smooth_x,
                    smooth_y,
                    color="#c23b22",
                    linewidth=2,
                    label=f"spline (bin n >= {min_bin_count})",
                )
                raw_axis.legend(loc="best", fontsize=8)

            for position, row in summary_df.iterrows():
                binned_axis.annotate(
                    f"n={int(row['bin_count'])}",
                    (position, row["mean_ci_high"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
            binned_axis.set(
                title=f"{target_column}: quantile-bin summary",
                xlabel=f"{feature_column} range within bin",
                ylabel="count target",
                xticks=positions,
                xticklabels=summary_df["feature_range"],
            )
            binned_axis.tick_params(axis="x", labelrotation=45, labelsize=8)
            binned_axis.legend(loc="best", fontsize=8)

        figure.suptitle(f"Marginal relationship: {feature_column}", fontsize=14)
        return figure, pd.concat(summaries, ignore_index=True)


    return (create_marginal_nonlinearity_figure,)


@app.cell
def _(
    cohort_target_columns,
    create_marginal_nonlinearity_figure,
    eda_final_df,
    marginal_support_threshold,
    mo,
    pd,
    plt,
):
    nonlinearity_feature_columns = [
        "ses",
        "n_daycares_500m",
        "avg_household_size",
        "median_age",
        "n_apartments",
    ]
    nonlinearity_target_columns = ["n_children_total", *cohort_target_columns]
    marginal_nonlinearity_figures: dict[str, plt.Figure] = {}
    _marginal_summaries: list[pd.DataFrame] = []

    for _feature_column in nonlinearity_feature_columns:
        _summary_figure, _summary_df = create_marginal_nonlinearity_figure(
            eda_final_df,
            _feature_column,
            nonlinearity_target_columns,
            min_bin_count=marginal_support_threshold,
        )
        marginal_nonlinearity_figures[_feature_column] = _summary_figure
        _marginal_summaries.append(_summary_df)

    marginal_nonlinearity_summary_df = pd.concat(
        _marginal_summaries,
        ignore_index=True,
    )

    mo.vstack(
        [
            mo.md(
                "### Quantile-bin support register\n"
                "Each row is an observed feature range. The `supported` flag "
                f"controls whether that bin contributes to the spline overlay "
                f"(n >= {marginal_support_threshold})."
            ),
            mo.ui.table(
                marginal_nonlinearity_summary_df.round(3),
                selection=None,
                pagination=True,
                page_size=12,
            ),
            *[
                mo.vstack(
                    [
                        mo.md(f"### {_feature_column}"),
                        marginal_nonlinearity_figures[_feature_column],
                    ]
                )
                for _feature_column in nonlinearity_feature_columns
            ],
        ]
    )
    return (
        marginal_nonlinearity_summary_df,
        nonlinearity_feature_columns,
        nonlinearity_target_columns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 8) Step 5: Exploratory Conditional Nonlinearities

    One histogram gradient-boosting Poisson model is fit per count target using
    prediction-time features only. These models are used solely to visualize
    flexible conditional means: they are not selected prediction models, formal
    evaluations, distributional uncertainty models, or causal estimators.

    Partial dependence averages predictions after replacing one feature. Because
    SES, daycare availability, household size, and median age are correlated, some
    replacements may have weak joint support. The plots therefore use the central
    observed feature range, include an observed-support rug, and should be read
    alongside the raw and binned Step 4 views.
    """)
    return


@app.cell
def _(
    eda_config,
    eda_final_df,
    nonlinearity_target_columns,
    numeric_feature_columns,
    pd,
):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence

    conditional_model_df = eda_final_df[numeric_feature_columns].astype(float).copy()
    _school_indicator_df = pd.get_dummies(
        eda_final_df["school_status"],
        prefix="school_status",
        drop_first=True,
        dtype=float,
    )
    conditional_model_df = pd.concat(
        [conditional_model_df, _school_indicator_df],
        axis=1,
    )
    conditional_model_feature_columns = conditional_model_df.columns.tolist()
    conditional_models: dict[str, HistGradientBoostingRegressor] = {}

    for _target_column in nonlinearity_target_columns:
        _conditional_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.5,
            random_state=eda_config.simulation.seed,
        )
        conditional_models[_target_column] = _conditional_model.fit(
            conditional_model_df,
            eda_final_df[_target_column],
        )

    conditional_model_register_df = pd.DataFrame(
        {
            "target": nonlinearity_target_columns,
            "estimator": "HistGradientBoostingRegressor",
            "loss": "poisson",
            "rows": len(conditional_model_df),
            "prediction_features": len(conditional_model_feature_columns),
            "role": "exploratory conditional-mean visualization only",
        }
    )
    conditional_model_register_df
    return (
        HistGradientBoostingRegressor,
        conditional_model_df,
        conditional_model_register_df,
        conditional_models,
        partial_dependence,
    )


@app.cell
def _(
    HistGradientBoostingRegressor,
    eda_config,
    np,
    partial_dependence,
    pd,
    plt,
):
    def create_pd_ice_figure(
        model: HistGradientBoostingRegressor,
        feature_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        ice_lines: int = 50,
        grid_resolution: int = 25,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """Create central-range PD/ICE panels and return the plotted PD values."""
        figure, figure_axes = plt.subplots(
            len(feature_columns),
            1,
            figsize=(11, 3.2 * len(feature_columns)),
            layout="constrained",
            squeeze=False,
        )
        rng = np.random.default_rng(eda_config.simulation.seed)
        curve_summaries: list[pd.DataFrame] = []

        for row_index, feature_column in enumerate(feature_columns):
            axis = figure_axes[row_index, 0]
            observed_values = feature_df[feature_column].to_numpy(dtype=float)
            central_bounds = np.quantile(observed_values, [0.05, 0.95])
            central_values = np.unique(
                observed_values[
                    (observed_values >= central_bounds[0])
                    & (observed_values <= central_bounds[1])
                ]
            )
            grid_values_for_feature = (
                central_values
                if len(central_values) <= grid_resolution
                else np.linspace(
                    float(central_bounds[0]),
                    float(central_bounds[1]),
                    grid_resolution,
                )
            )
            dependence = partial_dependence(
                model,
                feature_df,
                [feature_column],
                kind="both",
                method="brute",
                custom_values={feature_column: grid_values_for_feature},
            )
            grid_values = dependence["grid_values"][0]
            average_values = dependence["average"][0]
            individual_values = dependence["individual"][0]
            curve_summaries.append(
                pd.DataFrame(
                    {
                        "target": target_column,
                        "feature": feature_column,
                        "grid_order": np.arange(len(grid_values)),
                        "feature_value": grid_values,
                        "partial_dependence_mean": average_values,
                        "support_low": float(central_bounds[0]),
                        "support_high": float(central_bounds[1]),
                    }
                )
            )
            selected_rows = rng.choice(
                individual_values.shape[0],
                size=min(ice_lines, individual_values.shape[0]),
                replace=False,
            )
            axis.plot(
                grid_values,
                individual_values[selected_rows].T,
                color="0.55",
                alpha=0.12,
                linewidth=0.8,
            )
            axis.plot(
                grid_values,
                average_values,
                color="#b23a2b",
                linewidth=2.5,
                label="partial dependence",
            )
            support_values = feature_df[feature_column].sample(
                n=min(180, len(feature_df)),
                random_state=eda_config.simulation.seed,
            )
            axis.plot(
                support_values,
                np.zeros(len(support_values)),
                "|",
                color="#1f5a85",
                alpha=0.25,
                markersize=7,
                transform=axis.get_xaxis_transform(),
                label="observed support",
            )
            axis.set(
                title=f"{target_column}: {feature_column}",
                xlabel=feature_column,
                ylabel="predicted conditional mean",
            )
            axis.legend(loc="best", fontsize=8)

        figure.suptitle(
            f"Exploratory partial dependence and ICE: {target_column}",
            fontsize=14,
        )
        return figure, pd.concat(curve_summaries, ignore_index=True)

    return (create_pd_ice_figure,)


@app.cell
def _(
    conditional_model_df,
    conditional_model_register_df,
    conditional_models: "dict[str, HistGradientBoostingRegressor]",
    create_pd_ice_figure,
    mo,
    nonlinearity_feature_columns,
    nonlinearity_target_columns,
    pd,
    plt,
):
    conditional_effect_figures: dict[str, plt.Figure] = {}
    _conditional_summaries: list[pd.DataFrame] = []

    for _target_column in nonlinearity_target_columns:
        _conditional_figure, _conditional_summary_df = create_pd_ice_figure(
            conditional_models[_target_column],
            conditional_model_df,
            nonlinearity_feature_columns,
            _target_column,
        )
        conditional_effect_figures[_target_column] = _conditional_figure
        _conditional_summaries.append(_conditional_summary_df)

    conditional_effect_summary_df = pd.concat(
        _conditional_summaries,
        ignore_index=True,
    )

    mo.vstack(
        [
            mo.md("### Exploratory model register"),
            conditional_model_register_df,
            mo.callout(
                mo.md(
                    "**Interpretation limit:** Partial dependence averages model "
                    "predictions after replacing one feature. Correlated context "
                    "means some replacements have weak joint support. ICE "
                    "variation shows predictive heterogeneity, not causal effects "
                    "or uncertainty intervals."
                ),
                kind="warn",
            ),
            *[
                conditional_effect_figures[_target_column]
                for _target_column in nonlinearity_target_columns
            ],
        ]
    )
    return (conditional_effect_summary_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9) Step 6: Empirical Interaction And Composition Views

    These views stay on observed data. Curves compare binned target means across
    context strata, and heatmaps show the same cells with sparse combinations
    masked. Vertical separation alone can reflect a modifier main effect;
    descriptive interaction candidates require supported trajectories to differ
    in direction or shape. These are not causal effects or formal interaction
    tests.
    """)
    return


@app.cell
def _(eda_final_df, nonlinearity_target_columns, pd):
    interaction_support_threshold = 15


    def summarize_empirical_interaction(
        data_df: pd.DataFrame,
        x_column: str,
        modifier_column: str,
        targets: list[str],
        min_cell_count: int,
        x_bins: int = 8,
        modifier_bins: int = 4,
    ) -> pd.DataFrame:
        """Summarize observed target means over supported two-feature cells."""
        analysis_df = data_df[[x_column, modifier_column, *targets]].copy()
        if analysis_df[x_column].nunique() <= x_bins:
            analysis_df["x_bin"] = analysis_df[x_column]
        else:
            analysis_df["x_bin"] = pd.qcut(
                analysis_df[x_column],
                q=x_bins,
                duplicates="drop",
            )
        analysis_df["modifier_bin"] = pd.qcut(
            analysis_df[modifier_column],
            q=modifier_bins,
            duplicates="drop",
        )

        summary_df = (
            analysis_df.groupby(["x_bin", "modifier_bin"], observed=True)
            .agg(
                x_center=(x_column, "mean"),
                modifier_center=(modifier_column, "mean"),
                n_buildings=(x_column, "size"),
                **{f"{target}_mean": (target, "mean") for target in targets},
            )
            .reset_index()
        )
        summary_df["x_range"] = summary_df["x_bin"].astype(str)
        summary_df["modifier_range"] = summary_df["modifier_bin"].astype(str)
        summary_df["supported"] = summary_df["n_buildings"] >= min_cell_count
        return summary_df

    interaction_pair_specs = {
        "daycare_by_median_age": ("n_daycares_500m", "median_age", 9),
        "ses_by_household_size": ("ses", "avg_household_size", 8),
    }
    interaction_summary_frames = {
        pair_name: summarize_empirical_interaction(
            eda_final_df,
            x_column,
            modifier_column,
            nonlinearity_target_columns,
            min_cell_count=interaction_support_threshold,
            x_bins=x_bins,
        )
        for pair_name, (
            x_column,
            modifier_column,
            x_bins,
        ) in interaction_pair_specs.items()
    }
    interaction_support_df = pd.DataFrame(
        [
            {
                "interaction": pair_name,
                "observed_cells": len(summary_df),
                "supported_cells": int(summary_df["supported"].sum()),
                "sparse_cells": int((~summary_df["supported"]).sum()),
                "minimum_cell_count": int(summary_df["n_buildings"].min()),
            }
            for pair_name, summary_df in interaction_summary_frames.items()
        ]
    )
    interaction_support_df
    return (
        interaction_pair_specs,
        interaction_summary_frames,
        interaction_support_df,
        interaction_support_threshold,
    )


@app.cell
def _(np, pd, plt, sns):
    def create_interaction_curve_figure(
        summary_df: pd.DataFrame,
        x_column: str,
        modifier_column: str,
        targets: list[str],
    ) -> plt.Figure:
        """Plot empirical target curves across observed modifier strata."""
        figure, figure_axes = plt.subplots(
            2,
            2,
            figsize=(17, 9),
        )
        figure.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.08,
            top=0.80,
            hspace=0.30,
            wspace=0.24,
        )
        modifier_ranges = (
            summary_df.groupby("modifier_range", observed=True)["modifier_center"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(modifier_ranges)))

        for axis, target_column in zip(figure_axes.flat, targets, strict=True):
            for color, modifier_range in zip(colors, modifier_ranges, strict=True):
                stratum_df = summary_df[
                    summary_df["modifier_range"] == modifier_range
                ].sort_values("x_center")
                supported_df = stratum_df[stratum_df["supported"]]
                sparse_df = stratum_df[~stratum_df["supported"]]
                axis.plot(
                    supported_df["x_center"],
                    supported_df[f"{target_column}_mean"],
                    marker="o",
                    color=color,
                    linewidth=1.8,
                    label=f"{modifier_column}: {modifier_range}",
                )
                if not sparse_df.empty:
                    axis.scatter(
                        sparse_df["x_center"],
                        sparse_df[f"{target_column}_mean"],
                        facecolors="none",
                        edgecolors="#b23a2b",
                        linewidths=1.5,
                        s=55,
                    )
            x_label = (
                f"{x_column} (observed value)"
                if summary_df["x_range"].nunique() == summary_df["x_center"].nunique()
                else f"mean {x_column} within bin"
            )
            axis.set(
                title=target_column,
                xlabel=x_label,
                ylabel="observed mean target count",
            )
            axis.tick_params(axis="both", labelsize=11)
            axis.title.set_fontsize(13)
            axis.xaxis.label.set_size(12)
            axis.yaxis.label.set_size(12)
            axis.grid(alpha=0.2)

        legend_handles, legend_labels = figure_axes.flat[0].get_legend_handles_labels()
        figure.legend(
            legend_handles,
            legend_labels,
            title=f"{modifier_column} quartile",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=2,
            frameon=True,
            fontsize=11,
            title_fontsize=12,
        )
        figure.suptitle(
            f"Empirical interaction curves: {x_column} by {modifier_column}",
            fontsize=16,
            y=0.985,
        )
        return figure


    def create_interaction_heatmap_figure(
        summary_frames: dict[str, pd.DataFrame],
        pair_specs: dict[str, tuple[str, str, int]],
        target_column: str,
        min_cell_count: int,
    ) -> plt.Figure:
        """Plot supported empirical interaction cells with means and counts."""
        figure, figure_axes = plt.subplots(
            1,
            len(pair_specs),
            figsize=(16, 5.5),
            layout="constrained",
            squeeze=False,
        )

        for axis, (pair_name, (x_column, modifier_column, _)) in zip(
            figure_axes.flat,
            pair_specs.items(),
            strict=True,
        ):
            summary_df = summary_frames[pair_name]
            x_order = (
                summary_df.groupby("x_range", observed=True)["x_center"]
                .mean()
                .sort_values()
                .index
            )
            modifier_order = (
                summary_df.groupby("modifier_range", observed=True)["modifier_center"]
                .mean()
                .sort_values()
                .index
            )
            mean_table = summary_df.pivot(
                index="modifier_range",
                columns="x_range",
                values=f"{target_column}_mean",
            ).reindex(index=modifier_order, columns=x_order)
            count_table = summary_df.pivot(
                index="modifier_range",
                columns="x_range",
                values="n_buildings",
            ).reindex(index=mean_table.index, columns=mean_table.columns)
            support_table = summary_df.pivot(
                index="modifier_range",
                columns="x_range",
                values="supported",
            ).reindex(index=mean_table.index, columns=mean_table.columns)
            annotation_table = mean_table.copy().astype(object)
            for row_label in mean_table.index:
                for column_label in mean_table.columns:
                    cell_mean = mean_table.loc[row_label, column_label]
                    cell_count = count_table.loc[row_label, column_label]
                    annotation_table.loc[row_label, column_label] = (
                        ""
                        if pd.isna(cell_mean) or not bool(support_table.loc[row_label, column_label])
                        else f"{cell_mean:.1f}\n(n={int(cell_count)})"
                    )
            sns.heatmap(
                mean_table,
                mask=~support_table.fillna(False).astype(bool),
                annot=annotation_table,
                fmt="",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={"label": "observed mean count"},
                ax=axis,
            )
            axis.set(
                title=(
                    f"{x_column} by {modifier_column}\n"
                    f"masked where n < {min_cell_count}"
                ),
                xlabel=x_column,
                ylabel=modifier_column,
            )
            axis.tick_params(axis="x", rotation=45)
            axis.tick_params(axis="y", rotation=0)

        figure.suptitle(f"Empirical interaction heatmaps: {target_column}", fontsize=14)
        return figure

    return create_interaction_curve_figure, create_interaction_heatmap_figure


@app.cell
def _(
    create_interaction_curve_figure,
    create_interaction_heatmap_figure,
    interaction_pair_specs,
    interaction_summary_frames,
    interaction_support_df,
    interaction_support_threshold,
    mo,
    nonlinearity_target_columns,
):
    interaction_curve_figures = {
        pair_name: create_interaction_curve_figure(
            interaction_summary_frames[pair_name],
            x_column,
            modifier_column,
            nonlinearity_target_columns,
        )
        for pair_name, (
            x_column,
            modifier_column,
            _,
        ) in interaction_pair_specs.items()
    }
    interaction_heatmap_figures = {
        target_column: create_interaction_heatmap_figure(
            interaction_summary_frames,
            interaction_pair_specs,
            target_column,
            min_cell_count=interaction_support_threshold,
        )
        for target_column in nonlinearity_target_columns
    }

    mo.vstack(
        [
            mo.md(
                f"""### Faceted empirical target curves

    **How to read these figures**

    - Each panel is a different target count.
    - The horizontal axis varies the first feature: exact observed daycare counts,
      or the mean SES value within each of eight SES bins.
    - Each colored line restricts the data to one quartile of the second feature:
      median age for the daycare figure, and average household size for the SES
      figure.
    - Each point is the **observed mean target** among buildings in that joint
      group. Connecting points helps compare shapes; it does not estimate a
      continuous or causal effect.
    - Vertical separation alone can reflect a modifier main effect. A descriptive
      interaction candidate requires the left-to-right direction or shape to differ
      across strata. Hollow points have fewer than {interaction_support_threshold} buildings and should not be
      interpreted.
    """
            ),
            *[
                interaction_curve_figures[pair_name]
                for pair_name in interaction_pair_specs
            ],
            mo.md(
                f"""### Supported-cell heatmaps

    **How to read these heatmaps**

    - Each figure corresponds to one target. Within it, the left heatmap shows
      daycare count by median-age quartile, and the right heatmap shows SES bin by
      household-size quartile.
    - Columns move from lower to higher values of the first feature; rows move from
      lower to higher quartiles of the second feature.
    - Cell color and the first annotation number both represent the **observed mean
      target count** for buildings in that joint group. The second line, `n=...`,
      is the number of buildings supporting that mean.
    - A left-to-right color change repeated similarly in every row suggests mainly
      a relationship with the column feature. A top-to-bottom change repeated
      across columns suggests mainly a relationship with the row feature.
    - Look for an **interaction candidate** when the left-to-right pattern changes
      across rows: for example, the target rises with daycare in one age quartile
      but stays flat or falls in another. Isolated dark or light cells are less
      persuasive than a coherent pattern across neighboring supported cells.
    - Blank cells are masked because they contain fewer than {interaction_support_threshold} buildings. Do not
      infer a pattern through these unsupported combinations.

    These are descriptive group averages, not adjusted or causal effects. Use the
    faceted curves above to see the same cells as trajectories and the heatmaps to
    compare the full two-feature pattern at a glance.
    """
            ),
            interaction_support_df,
            *[
                interaction_heatmap_figures[target_column]
                for target_column in nonlinearity_target_columns
            ],
            mo.callout(
                mo.md(
                    f"Hollow curve markers and masked heatmap cells have fewer "
                    f"than {interaction_support_threshold} buildings. Stratum "
                    "differences are interaction "
                    "candidates only; no formal interaction test is performed."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(create_share_df, eda_final_df, mo, np, pd, plt, room_columns):
    _room_share_df = create_share_df(
        df=eda_final_df,
        share_columns=room_columns,
        total_column="n_apartments",
        mask=eda_final_df["n_apartments"] > 0,
    ).rename(
        columns={
            _room_column: f"{_room_column}_share"
            for _room_column in room_columns
        }
    )
    _room_share_columns = _room_share_df.columns.tolist()
    _room_composition_df = pd.concat(
        [eda_final_df[["n_apartments"]], _room_share_df],
        axis=1,
    )
    _room_composition_df["apartment_bin"] = pd.qcut(
        _room_composition_df["n_apartments"],
        q=8,
        duplicates="drop",
    )
    room_composition_summary_df = (
        _room_composition_df.groupby("apartment_bin", observed=True)
        .agg(
            n_apartments_mean=("n_apartments", "mean"),
            n_buildings=("n_apartments", "size"),
            **{
                f"{_room_column}_share": (f"{_room_column}_share", "mean")
                for _room_column in room_columns
            },
        )
        .reset_index()
    )
    room_composition_summary_df["apartment_range"] = (
        room_composition_summary_df["apartment_bin"].astype(str)
    )

    room_composition_figure, _room_axis = plt.subplots(
        figsize=(10, 5.5),
        layout="constrained",
    )
    _room_colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(room_columns)))
    for _color, _room_column in zip(_room_colors, room_columns, strict=True):
        _room_axis.plot(
            room_composition_summary_df["n_apartments_mean"],
            room_composition_summary_df[f"{_room_column}_share"],
            marker="o",
            linewidth=2,
            color=_color,
            label=_room_column,
        )
    _room_axis.set(
        title="Room composition across building-size bins",
        xlabel="mean apartments per building within size bin",
        ylabel="mean share of apartments",
    )
    _room_axis.set_ylim(
        0,
        1.15 * room_composition_summary_df[_room_share_columns].to_numpy().max(),
    )
    _room_axis.legend(title="Room category", ncols=2)
    _room_axis.grid(alpha=0.2)

    mo.vstack(
        [
            mo.md("### Room composition and building size"),
            room_composition_figure,
            mo.md(
                "Each line shows the mean within-building share for one room "
                "category across eight quantile bins of `n_apartments`. The table "
                "reports the building count and mean values behind each point."
            ),
            room_composition_summary_df.drop(columns="apartment_bin"),
            mo.md(
                "Room counts sum exactly to `n_apartments`, so this comparison uses "
                "within-building shares rather than raw room counts."
            ),
        ]
    )
    return (room_composition_summary_df,)


@app.cell
def _(
    conditional_effect_summary_df,
    interaction_pair_specs,
    interaction_summary_frames,
    marginal_nonlinearity_summary_df,
    nonlinearity_target_columns,
    np,
    pd,
):
    def _classify_ses_curve(curve_df: pd.DataFrame, x_column: str, y_column: str) -> dict:
        ordered_df = curve_df.sort_values(x_column).reset_index(drop=True)
        x_values = ordered_df[x_column].to_numpy(dtype=float)
        y_values = ordered_df[y_column].to_numpy(dtype=float)
        result = {
            "classification": "insufficient points",
            "start_value": np.nan,
            "end_value": np.nan,
            "minimum_value": np.nan,
            "minimum_location": np.nan,
            "minimum_is_interior": False,
            "maximum_value": np.nan,
            "maximum_location": np.nan,
            "maximum_is_interior": False,
        }
        if len(y_values) < 3:
            return result

        minimum_index = int(np.argmin(y_values))
        maximum_index = int(np.argmax(y_values))
        minimum_is_interior = 0 < minimum_index < len(y_values) - 1
        maximum_is_interior = 0 < maximum_index < len(y_values) - 1
        is_u_shaped = minimum_is_interior and (
            y_values[0] > y_values[minimum_index]
            and y_values[-1] > y_values[minimum_index]
        )
        is_inverted_u = maximum_is_interior and (
            y_values[0] < y_values[maximum_index]
            and y_values[-1] < y_values[maximum_index]
        )
        increments = np.diff(y_values)
        if is_u_shaped and is_inverted_u:
            classification = "ambiguous: multiple interior extrema"
        elif is_u_shaped:
            classification = "U-shaped candidate"
        elif is_inverted_u:
            classification = "inverted-U candidate"
        elif np.all(increments >= 0) and np.any(increments > 0):
            classification = "monotone increasing"
        elif np.all(increments <= 0) and np.any(increments < 0):
            classification = "monotone decreasing"
        elif np.all(increments == 0):
            classification = "flat"
        else:
            classification = "ambiguous/non-monotone"

        return {
            "classification": classification,
            "start_value": float(y_values[0]),
            "end_value": float(y_values[-1]),
            "minimum_value": float(y_values[minimum_index]),
            "minimum_location": float(x_values[minimum_index]),
            "minimum_is_interior": minimum_is_interior,
            "maximum_value": float(y_values[maximum_index]),
            "maximum_location": float(x_values[maximum_index]),
            "maximum_is_interior": maximum_is_interior,
        }


    def _classify_daycare_curve(curve_df: pd.DataFrame, x_column: str, y_column: str) -> dict:
        ordered_df = curve_df.sort_values(x_column).drop_duplicates(x_column)
        x_values = ordered_df[x_column].to_numpy(dtype=float)
        y_values = ordered_df[y_column].to_numpy(dtype=float)
        result = {
            "classification": "insufficient increments",
            "early_mean_increment": np.nan,
            "late_mean_increment": np.nan,
            "direction_consistent": False,
            "increment_count": max(len(y_values) - 1, 0),
        }
        if len(y_values) < 3:
            return result

        increments = np.diff(y_values) / np.diff(x_values)
        split_index = len(increments) // 2
        if split_index == 0 or split_index == len(increments):
            return result
        early_increment = float(np.mean(increments[:split_index]))
        late_increment = float(np.mean(increments[split_index:]))
        nondecreasing = bool(np.all(increments >= 0) and np.any(increments > 0))
        nonincreasing = bool(np.all(increments <= 0) and np.any(increments < 0))
        direction_consistent = nondecreasing or nonincreasing or bool(np.all(increments == 0))
        if nondecreasing and abs(late_increment) < abs(early_increment):
            classification = "diminishing-returns candidate"
        elif nondecreasing and np.isclose(abs(late_increment), abs(early_increment)):
            classification = "approximately constant increments"
        elif nondecreasing:
            classification = "accelerating increments"
        elif nonincreasing:
            classification = "consistently decreasing"
        elif np.all(increments == 0):
            classification = "flat"
        else:
            classification = "ambiguous/non-monotone"
        return {
            "classification": classification,
            "early_mean_increment": early_increment,
            "late_mean_increment": late_increment,
            "direction_consistent": direction_consistent,
            "increment_count": len(increments),
        }


    _nonlinearity_rows = []
    for _feature_column in ["ses", "n_daycares_500m"]:
        for _target_column in nonlinearity_target_columns:
            _conditional_curve_df = conditional_effect_summary_df.loc[
                (conditional_effect_summary_df["feature"] == _feature_column)
                & (conditional_effect_summary_df["target"] == _target_column)
            ]
            _marginal_curve_df = marginal_nonlinearity_summary_df.loc[
                (marginal_nonlinearity_summary_df["feature"] == _feature_column)
                & (marginal_nonlinearity_summary_df["target"] == _target_column)
                & marginal_nonlinearity_summary_df["supported"]
            ]
            if _feature_column == "ses":
                _conditional_evidence = _classify_ses_curve(
                    _conditional_curve_df, "feature_value", "partial_dependence_mean"
                )
                _marginal_evidence = _classify_ses_curve(
                    _marginal_curve_df, "feature_center", "target_mean"
                )
            else:
                _conditional_evidence = _classify_daycare_curve(
                    _conditional_curve_df, "feature_value", "partial_dependence_mean"
                )
                _marginal_evidence = _classify_daycare_curve(
                    _marginal_curve_df, "feature_center", "target_mean"
                )
            _nonlinearity_rows.append(
                {
                    "feature": _feature_column,
                    "target": _target_column,
                    "conditional_classification": _conditional_evidence["classification"],
                    "marginal_classification": _marginal_evidence["classification"],
                    "conditional_marginal_agreement": (
                        _conditional_evidence["classification"]
                        == _marginal_evidence["classification"]
                    ),
                    **{
                        f"conditional_{_key}": _value
                        for _key, _value in _conditional_evidence.items()
                        if _key != "classification"
                    },
                    **{
                        f"marginal_{_key}": _value
                        for _key, _value in _marginal_evidence.items()
                        if _key != "classification"
                    },
                    "limitation": (
                        "Conditional PD is model-dependent; marginal disagreement "
                        "indicates confounding or model dependence."
                    ),
                }
            )
    nonlinearity_evidence_df = pd.DataFrame(_nonlinearity_rows)


    def _change_sign(value: float) -> str:
        if value > 0:
            return "+"
        if value < 0:
            return "-"
        return "0"


    _interaction_rows = []
    for _pair_name, (_x_column, _modifier_column, _x_bins) in interaction_pair_specs.items():
        _pair_df = interaction_summary_frames[_pair_name]
        _modifier_order = (
            _pair_df.groupby("modifier_range", observed=True)["modifier_center"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        _supported_df = _pair_df.loc[_pair_df["supported"]].copy()
        _support_counts = _supported_df.groupby("x_range", observed=True)[
            "modifier_range"
        ].nunique()
        _common_x_labels = _support_counts.loc[
            _support_counts == len(_modifier_order)
        ].index.tolist()
        _common_x_order = (
            _pair_df.loc[_pair_df["x_range"].isin(_common_x_labels)]
            .groupby("x_range", observed=True)["x_center"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        for _target_column in nonlinearity_target_columns:
            _target_mean_column = f"{_target_column}_mean"
            _stratum_records = []
            for _modifier_range in _modifier_order:
                _trajectory_df = (
                    _supported_df.loc[
                        (_supported_df["modifier_range"] == _modifier_range)
                        & _supported_df["x_range"].isin(_common_x_order)
                    ]
                    .set_index("x_range")
                    .reindex(_common_x_order)
                )
                _trajectory_values = _trajectory_df[_target_mean_column].to_numpy(
                    dtype=float
                )
                if len(_trajectory_values) >= 2 and np.isfinite(_trajectory_values).all():
                    _endpoint_change = float(
                        _trajectory_values[-1] - _trajectory_values[0]
                    )
                    _trajectory_signature = "".join(
                        _change_sign(float(_change))
                        for _change in np.diff(_trajectory_values)
                    )
                    _endpoint_direction = _change_sign(_endpoint_change)
                else:
                    _endpoint_change = np.nan
                    _trajectory_signature = "unavailable"
                    _endpoint_direction = "unavailable"
                _stratum_records.append(
                    {
                        "modifier_range": _modifier_range,
                        "endpoint_change": _endpoint_change,
                        "endpoint_direction": _endpoint_direction,
                        "trajectory_signature": _trajectory_signature,
                    }
                )
            _available_records = [
                _record for _record in _stratum_records
                if _record["endpoint_direction"] != "unavailable"
            ]
            _directions = {
                _record["endpoint_direction"] for _record in _available_records
            }
            _signatures = {
                _record["trajectory_signature"] for _record in _available_records
            }
            if len(_common_x_order) < 2 or len(_available_records) < 2:
                _interaction_classification = "inconclusive: insufficient common support"
            elif len(_directions) > 1 or len(_signatures) > 1:
                _interaction_classification = "descriptive interaction candidate"
            else:
                _interaction_classification = "no direction/shape-sign difference"
            for _record in _stratum_records:
                _interaction_rows.append(
                    {
                        "interaction": _pair_name,
                        "x_feature": _x_column,
                        "modifier": _modifier_column,
                        "target": _target_column,
                        **_record,
                        "classification": _interaction_classification,
                        "common_x_count": len(_common_x_order),
                        "common_x_start": (
                            _common_x_order[0] if _common_x_order else None
                        ),
                        "common_x_end": (
                            _common_x_order[-1] if _common_x_order else None
                        ),
                        "observed_cells": len(_pair_df),
                        "supported_cells": int(_pair_df["supported"].sum()),
                        "excluded_sparse_cells": int((~_pair_df["supported"]).sum()),
                        "limitation": (
                            "Descriptive supported-cell comparison only; common-range "
                            "restriction can exclude edge behavior."
                        ),
                    }
                )
    interaction_evidence_df = pd.DataFrame(_interaction_rows)
    return interaction_evidence_df, nonlinearity_evidence_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10) Findings And Next Steps

    The conclusions below are grouped by topic. Each bullet separates the observed
    evidence from the recommended next step and the main limitation.
    """)
    return


@app.cell
def _(
    conditional_effect_summary_df,
    conditional_model_register_df,
    eda_final_df,
    feature_target_corr_df,
    interaction_evidence_df,
    interaction_support_df,
    interaction_support_threshold,
    mo,
    nonlinearity_evidence_df,
    pd,
    quality_summary_df,
    room_columns,
    room_composition_summary_df,
    target_distribution_summary_df,
):
    _total_target = target_distribution_summary_df.set_index("target").loc[
        "n_children_total"
    ]
    _missing_value_count = int(eda_final_df.isna().sum().sum())
    _largest_absolute_correlation = float(feature_target_corr_df.abs().max().max())
    _room_share_ranges = {
        _room_column: float(
            room_composition_summary_df[f"{_room_column}_share"].max()
            - room_composition_summary_df[f"{_room_column}_share"].min()
        )
        for _room_column in room_columns
    }
    _largest_room_share_range = max(_room_share_ranges.values())
    _smallest_room_bin_count = int(room_composition_summary_df["n_buildings"].min())
    _sparse_interaction_cells = int(interaction_support_df["sparse_cells"].sum())
    _total_interaction_cells = int(interaction_support_df["observed_cells"].sum())

    _conditional_model_count = len(conditional_model_register_df)
    _conditional_panel_count = conditional_effect_summary_df.groupby(
        ["target", "feature"]
    ).ngroups
    _nonlinearity_summary = {
        _feature_name: {
            "conditional": "; ".join(
                _feature_df["target"].astype(str)
                + ": "
                + _feature_df["conditional_classification"].astype(str)
            ),
            "agreement_count": int(
                _feature_df["conditional_marginal_agreement"].sum()
            ),
        }
        for _feature_name, _feature_df in nonlinearity_evidence_df.groupby(
            "feature", sort=False
        )
    }
    _ses_nonlinearity_summary = _nonlinearity_summary["ses"]
    _daycare_nonlinearity_summary = _nonlinearity_summary["n_daycares_500m"]

    _interaction_target_df = interaction_evidence_df[
        ["interaction", "target", "classification"]
    ].drop_duplicates()
    _interaction_candidate_counts = (
        _interaction_target_df.assign(
            _is_candidate=_interaction_target_df["classification"].eq(
                "descriptive interaction candidate"
            )
        )
        .groupby("interaction", observed=True)["_is_candidate"]
        .sum()
    )
    _interaction_evidence_summary_df = interaction_evidence_df.groupby(
        "interaction", observed=True
    ).agg(
        endpoint_change_min=("endpoint_change", "min"),
        endpoint_change_max=("endpoint_change", "max"),
        common_x_count=("common_x_count", "first"),
        excluded_sparse_cells=("excluded_sparse_cells", "first"),
    )
    _daycare_interaction_summary = _interaction_evidence_summary_df.loc[
        "daycare_by_median_age"
    ]
    _ses_interaction_summary = _interaction_evidence_summary_df.loc[
        "ses_by_household_size"
    ]

    eda_findings_df = pd.DataFrame(
        [
            {
                "area": "data quality",
                "status": "established evidence",
                "finding": "Structural, accounting, and missingness checks pass",
                "evidence": (
                    f"{int(quality_summary_df['passed'].sum())}/"
                    f"{len(quality_summary_df)} checks passed; "
                    f"{_missing_value_count} missing values"
                ),
                "modeling_implication": (
                    "Use the audited prediction-time table as the later modeling base."
                ),
                "limitation": "Checks establish internal consistency, not external validity.",
            },
            {
                "area": "analysis population",
                "status": "established evidence",
                "finding": "Expanded deterministic EDA population generated",
                "evidence": (
                    f"{len(eda_final_df):,} buildings across "
                    f"{eda_final_df['neighborhood_id'].nunique()} neighborhoods"
                ),
                "modeling_implication": (
                    "Preserve neighborhood structure in later validation splits."
                ),
                "limitation": "The population is simulated rather than externally sampled.",
            },
            {
                "area": "count likelihood",
                "status": "established evidence",
                "finding": "Total-child counts are overdispersed",
                "evidence": f"variance/mean = {_total_target['variance_to_mean']:.2f}",
                "modeling_implication": (
                    "Use NB2 as the primary later likelihood and compare Poisson "
                    "through held-out count deviance and calibration."
                ),
                "limitation": "A marginal variance-to-mean ratio does not identify every source of dispersion.",
            },
            {
                "area": "room-count collinearity",
                "status": "established evidence",
                "finding": "Room counts are deterministic components of building size",
                "evidence": (
                    "The 3-6 room counts sum exactly to n_apartments for all "
                    f"{len(eda_final_df):,} buildings"
                ),
                "modeling_implication": (
                    "Use a composition-aware representation; do not include all room "
                    "counts with n_apartments without an identifiability strategy."
                ),
                "limitation": "This constraint does not determine the best composition parameterization.",
            },
            {
                "area": "room composition",
                "status": "descriptive evidence",
                "finding": "Mean room shares vary little across building-size bins",
                "evidence": (
                    f"largest mean-share range = {_largest_room_share_range:.3f} "
                    f"across 8 bins; minimum bin size = {_smallest_room_bin_count}"
                ),
                "modeling_implication": (
                    "Treat building size as distinct from room mix and validate whether "
                    "composition adds predictive value."
                ),
                "limitation": "Binned means can hide building-level composition variation.",
            },
            {
                "area": "linear association",
                "status": "descriptive evidence",
                "finding": "Pairwise correlations alone do not resolve functional form",
                "evidence": (
                    "largest absolute Pearson/Spearman feature-target coefficient = "
                    f"{_largest_absolute_correlation:.2f}"
                ),
                "modeling_implication": "Assess functional form conditionally and out of sample.",
                "limitation": "Pairwise coefficients omit nonlinear and joint relationships.",
            },
            {
                "area": "SES nonlinearity",
                "status": "inconclusive",
                "finding": "SES curvature is target-specific and not consistently supported",
                "evidence": (
                    f"Across {_conditional_model_count} exploratory models and "
                    f"{_conditional_panel_count} materialized PD panels, SES conditional "
                    f"PD: {_ses_nonlinearity_summary['conditional']}. Conditional and "
                    f"marginal classifications agree for "
                    f"{_ses_nonlinearity_summary['agreement_count']}/4 targets."
                ),
                "modeling_implication": (
                    "Compare a linear SES term with a low-degree quadratic or low-df "
                    "spline; select using held-out count deviance and calibration."
                ),
                "limitation": (
                    "PD is model-dependent, and marginal disagreement indicates "
                    "confounding or model dependence; no causal effect is identified."
                ),
            },
            {
                "area": "daycare nonlinearity",
                "status": "not visually supported",
                "finding": "Diminishing daycare returns are not supported by these views",
                "evidence": (
                    f"Across {_conditional_model_count} exploratory models and "
                    f"{_conditional_panel_count} materialized PD panels, daycare "
                    f"conditional PD: {_daycare_nonlinearity_summary['conditional']}. "
                    f"Conditional and marginal classifications agree for "
                    f"{_daycare_nonlinearity_summary['agreement_count']}/4 targets."
                ),
                "modeling_implication": (
                    "Do not prioritize a saturation transform. If revisited, estimate its "
                    "shape with a monotone transform or low-df spline rather than fixing it."
                ),
                "limitation": (
                    "PD is model-dependent, and strict increment consistency can classify "
                    "small reversals as ambiguous; no causal effect is identified."
                ),
            },
            {
                "area": "daycare by median-age interaction",
                "status": "supported descriptive candidate",
                "finding": "Supported daycare trajectories differ across age strata",
                "evidence": (
                    f"{int(_interaction_candidate_counts['daycare_by_median_age'])}/4 "
                    f"targets have differing endpoint "
                    f"directions or trajectory sign patterns over "
                    f"{int(_daycare_interaction_summary['common_x_count'])} common "
                    "daycare levels; "
                    f"endpoint changes range from "
                    f"{_daycare_interaction_summary['endpoint_change_min']:.2f} to "
                    f"{_daycare_interaction_summary['endpoint_change_max']:.2f}."
                ),
                "modeling_implication": (
                    "Test a daycare-by-age product or smooth interaction later; retain it "
                    "only if stable across resamples and beneficial out of sample."
                ),
                "limitation": (
                    f"Descriptive unadjusted means only; "
                    f"{int(_daycare_interaction_summary['excluded_sparse_cells'])} sparse "
                    "cells are excluded and common-range restriction omits edge behavior."
                ),
            },
            {
                "area": "SES by household-size interaction",
                "status": "supported descriptive candidate",
                "finding": "Supported SES trajectories differ across household-size strata",
                "evidence": (
                    f"{int(_interaction_candidate_counts['ses_by_household_size'])}/4 "
                    f"targets have differing endpoint "
                    f"directions or trajectory sign patterns over "
                    f"{int(_ses_interaction_summary['common_x_count'])} common SES "
                    "bins; "
                    f"endpoint changes range from "
                    f"{_ses_interaction_summary['endpoint_change_min']:.2f} to "
                    f"{_ses_interaction_summary['endpoint_change_max']:.2f}."
                ),
                "modeling_implication": (
                    "Test an SES-by-household-size product or smooth interaction later; "
                    "retain it only if stable across resamples and beneficial out of sample."
                ),
                "limitation": (
                    "Descriptive unadjusted means only; trajectory sign patterns can be "
                    "noise-sensitive and no formal interaction test is performed."
                ),
            },
            {
                "area": "unsupported interaction regions",
                "status": "established evidence",
                "finding": "Sparse interaction cells are excluded from interpretation",
                "evidence": (
                    f"{_sparse_interaction_cells}/{_total_interaction_cells} observed "
                    f"cells contain fewer than {interaction_support_threshold} buildings "
                    "and are masked"
                ),
                "modeling_implication": (
                    "Avoid interpreting or validating interaction behavior in unsupported regions."
                ),
                "limitation": "Support thresholds reduce the range covered by descriptive comparisons.",
            },
        ]
    )
    _finding_topics = {
        "Data Readiness": [
            "data quality",
            "analysis population",
        ],
        "Count Outcomes And Validation": [
            "count likelihood",
            "linear association",
        ],
        "Building Size And Room Composition": [
            "room-count collinearity",
            "room composition",
        ],
        "Functional Form": [
            "SES nonlinearity",
            "daycare nonlinearity",
        ],
        "Interactions And Support": [
            "daycare by median-age interaction",
            "SES by household-size interaction",
            "unsupported interaction regions",
        ],
    }
    _findings_indexed_df = eda_findings_df.set_index("area")
    _finding_sections = []
    for _topic, _areas in _finding_topics.items():
        _topic_lines = [f"### {_topic}"]
        for _area in _areas:
            _finding_row = _findings_indexed_df.loc[_area]
            _topic_lines.extend(
                [
                    "",
                    f"- **{_finding_row['finding']}** "
                    f"*({_finding_row['status']})*  ",
                    f"  **Evidence:** {_finding_row['evidence']}  ",
                    f"  **Next step:** {_finding_row['modeling_implication']}  ",
                    f"  **Caveat:** {_finding_row['limitation']}",
                ]
            )
        _finding_sections.append("\n".join(_topic_lines))

    mo.md("\n\n".join(_finding_sections))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11) Step 7: Modeling Handoff And Split Design

    This step documents the feature representation and validation design for later
    model fitting. It deliberately does **not** create a train/test split or fit a
    model.
    """)
    return


@app.cell
def _(eda_config, mo, pd):
    modeling_split_seed = int(eda_config.simulation.seed)
    modeling_test_fraction = 0.2
    modeling_split_strategy = "known-neighborhood randomized within neighborhood"

    modeling_feature_columns = [
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
        "3_rooms_share",
        "4_rooms_share",
        "5_rooms_share",
        "school_status",
    ]

    modeling_feature_schema_df = pd.DataFrame(
        [
            {
                "feature": "ses",
                "type": "numeric context",
                "representation": "raw value; standardize inside each training fold for linear/probabilistic models",
                "functional_form": "linear baseline; compare a quadratic or low-df spline",
                "interaction_candidate": "ses x avg_household_size",
            },
            {
                "feature": "avg_household_size",
                "type": "numeric context",
                "representation": "raw value; standardize inside each training fold for linear/probabilistic models",
                "functional_form": "linear baseline",
                "interaction_candidate": "ses x avg_household_size",
            },
            {
                "feature": "median_age",
                "type": "numeric context",
                "representation": "raw value; standardize inside each training fold for linear/probabilistic models",
                "functional_form": "linear baseline",
                "interaction_candidate": "n_daycares_500m x median_age",
            },
            {
                "feature": "n_daycares_500m",
                "type": "numeric context",
                "representation": "raw count",
                "functional_form": "linear baseline; do not prioritize saturation; revisit a monotone transform or low-df spline only if validation supports it",
                "interaction_candidate": "n_daycares_500m x median_age",
            },
            {
                "feature": "n_apartments",
                "type": "numeric size",
                "representation": "raw count; use as size/exposure where the model supports it",
                "functional_form": "linear baseline",
                "interaction_candidate": "none proposed from EDA",
            },
            {
                "feature": "3_rooms_share",
                "type": "room composition",
                "representation": "share of apartments; use with n_apartments",
                "functional_form": "linear baseline",
                "interaction_candidate": "none proposed from EDA",
            },
            {
                "feature": "4_rooms_share",
                "type": "room composition",
                "representation": "share of apartments; use with n_apartments",
                "functional_form": "linear baseline",
                "interaction_candidate": "none proposed from EDA",
            },
            {
                "feature": "5_rooms_share",
                "type": "room composition",
                "representation": "share of apartments; use with n_apartments",
                "functional_form": "linear baseline",
                "interaction_candidate": "none proposed from EDA",
            },
            {
                "feature": "school_status",
                "type": "categorical",
                "representation": "one-hot encode inside each training fold; use none as the reference for statistical models",
                "functional_form": "category effects",
                "interaction_candidate": "none proposed from EDA",
            },
        ]
    )

    split_design = mo.md(
        f"""### Recommended validation design

    **Primary deployment claim: a new building in a known neighborhood.** When
    model fitting begins, use a reproducible `{modeling_test_fraction:.0%}` test
    holdout randomized *within each neighborhood* with seed
    `{modeling_split_seed}`. For every neighborhood with at least two buildings,
    shuffle its building rows using that seed and assign a bounded test count of
    `max(1, min(round(n * test_fraction), n - 1))`; retain singleton neighborhoods
    in training. Fit all preprocessing, encoders, scaling, transformations, and
    feature-selection decisions on the resulting training partition only. Keep the
    test partition untouched until the model and calibration choices are fixed.

    This split is appropriate because its claim is conditional prediction for a
    new building where the neighborhood context is already known. It must not be
    reported as unseen-neighborhood performance: both partitions contain buildings
    from the same neighborhoods.

    **Alternative claim: a new building in an unseen neighborhood.** Use repeated
    grouped folds or leave-one-neighborhood-out validation keyed by
    `neighborhood_id`. That design evaluates context transfer, has fewer effective
    groups, and should report the distribution of fold-level scores rather than a
    single aggregate metric.
    """
    )

    mo.vstack(
        [
            split_design,
            mo.md("### Modeling feature schema"),
            modeling_feature_schema_df,
            mo.callout(
                mo.md(
                    "The schema uses `n_apartments` plus three room shares and drops "
                    "`6_rooms_share`, avoiding the deterministic room-count identity. "
                    "The listed nonlinearities and interactions are EDA candidates, "
                    "not selected terms; retain them only when held-out validation and "
                    "resampling support them."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
