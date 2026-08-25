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
    # Building-Level Age-Group Modeling

    This notebook begins the modeling phase with the shared data contract and
    validation split. It currently:

    1. Reproduces the canonical population used by the EDA handoff.
    2. Constructs the agreed non-collinear room-share representation.
    3. Records the complete base predictor set and deferred candidate terms.
    4. Creates a deterministic building holdout within each neighborhood.

    No model is fit in this implementation slice. The primary validation claim
    is prediction for a new building in a known neighborhood; results from this
    split must not be described as unseen-neighborhood performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Setup And Canonical Population

    Use the same configuration and EDA-only population overrides as the
    completed exploratory analysis so the modeling handoff is reproducible.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path
    import sys

    import numpy as np
    import pandas as pd

    notebook_source_path = Path(mo.notebook_dir()).parent / "src"
    if str(notebook_source_path) not in sys.path:
        sys.path.insert(0, str(notebook_source_path))

    from student_simulator import (
        StudentPopulationSimulator,
        load_simulation_config,
    )

    return StudentPopulationSimulator, load_simulation_config, np, pd


@app.cell
def _(StudentPopulationSimulator, load_simulation_config, mo):
    project_root = mo.notebook_dir().parent
    config_path = project_root / "configs" / "stage1.toml"
    base_modeling_config = load_simulation_config(config_path)
    modeling_config = base_modeling_config.model_copy(
        update={
            "simulation": base_modeling_config.simulation.model_copy(
                update={"n_neighborhoods": 150}
            ),
            "building": base_modeling_config.building.model_copy(
                update={"buildings_per_neighborhood_rate": 9.0}
            ),
        }
    )
    modeling_simulator = StudentPopulationSimulator(modeling_config)
    modeling_source_df = modeling_simulator.run()
    mo.md(
        f"""
        **Canonical modeling population:** `{len(modeling_source_df):,}` buildings
        across `{modeling_source_df["neighborhood_id"].nunique()}` neighborhoods.

        - Configuration: `{config_path.relative_to(project_root)}`
        - Simulation seed: `{modeling_config.simulation.seed}`
        - Building rate: `{modeling_config.building.buildings_per_neighborhood_rate}`
        """
    )
    return modeling_config, modeling_source_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Modeling Data Contract

    The current base predictor set is complete for the simulator's exported
    prediction-time data. Building size is represented by `n_apartments`; room mix
    is represented by 3-, 4-, and 5-room shares, with the 6-room share omitted as
    the composition reference. Identifiers, targets, and private random effects are
    never model predictors.

    ### Deferred candidate terms

    Do not add these to the shared table yet. Construct them inside each training
    fold and retain them only when held-out improvement is stable:

    - **EDA priority:** compare linear `ses` with a quadratic or low-degree spline;
      test `ses` by `avg_household_size` and `n_daycares_500m` by `median_age`.
    - **Secondary sensitivity checks:** test room composition by
      `avg_household_size` for total counts and room composition by `median_age` for
      age-group probabilities. Because the generator mechanisms operate on
      apartment room counts, try a parsimonious building summary such as
      `mean_rooms` or `large_room_excess_share` as an alternative to interacting
      every share. Do not add redundant room representations together.
    - **Low priority:** revisit daycare with a monotone transform or low-degree
      spline only if residual diagnostics justify it. The EDA did not support
      copying the simulator's fixed saturation form by default.

    The simplified simulator generates room mix partly from `ses` and
    `avg_household_size`, so room-share coefficients are conditional associations,
    not independent effects. Its private building and neighborhood effects imply
    residual clustering and predictive uncertainty; assess these with
    neighborhood-aware validation rather than adding latent values as features.

    Standardization, centering, one-hot encoding, and `log(n_apartments)` as an
    exposure offset are preprocessing choices rather than additional predictors.
    They must be defined from the training partition only.
    """)
    return


@app.cell
def _():
    identifier_columns = ["building_id", "neighborhood_id"]

    numeric_features = [
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
    ]
    categorical_modeling_features = ["school_status"]

    cohort_target_columns = [
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    ]
    modeling_target_columns = [*cohort_target_columns, "n_children_total"]
    return (
        categorical_modeling_features,
        identifier_columns,
        modeling_target_columns,
    )


@app.cell
def _(pd):
    class DatasetEnricher:
        """Build a modeling copy with the agreed room-share representation."""

        def __init__(self, df: pd.DataFrame):
            self.df = df

        def add_room_shares(
            self,
            n_apartments_column: str,
            n_rooms_columns: list[str],
        ) -> pd.DataFrame:
            """Return a copy with three room shares and no source mutation."""
            required_columns = {n_apartments_column, *n_rooms_columns}
            missing_columns = required_columns.difference(self.df.columns)
            if missing_columns:
                raise ValueError(f"Missing room columns: {sorted(missing_columns)}")
            if (self.df[n_apartments_column] <= 0).any():
                raise ValueError(
                    f"{n_apartments_column} must be positive before computing shares"
                )

            prepared_df = self.df.copy()
            for room_column in sorted(n_rooms_columns)[:-1]:
                prepared_df[f"{room_column}_share"] = (
                    prepared_df[room_column] / prepared_df[n_apartments_column]
                )
            return prepared_df

    return (DatasetEnricher,)


@app.cell
def _(
    DatasetEnricher,
    categorical_modeling_features,
    identifier_columns,
    modeling_source_df,
    modeling_target_columns,
):
    dataset_enricher = DatasetEnricher(modeling_source_df)
    modeling_df = dataset_enricher.add_room_shares(
        n_apartments_column="n_apartments",
        n_rooms_columns=["3_rooms", "4_rooms", "5_rooms", "6_rooms"],
    )

    numeric_modeling_features = [
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
        "3_rooms_share",
        "4_rooms_share",
        "5_rooms_share",
    ]
    modeling_feature_columns = [
        *numeric_modeling_features,
        *categorical_modeling_features,
    ]

    modeling_df = modeling_df[identifier_columns + modeling_feature_columns + modeling_target_columns]
    modeling_df.head(8)
    return (modeling_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Known-Neighborhood Building Split

    Split before fitting preprocessing or constructing Model B's child-level
    categorical view. Within every neighborhood containing at least two
    buildings, assign

    $$
    n_{test}=\max\left(1,\min\left(\operatorname{round}(0.2n),n-1\right)\right).
    $$

    Singleton neighborhoods remain in training. Sorting building IDs before the
    seeded permutation makes the result invariant to input row order.
    """)
    return


@app.cell
def _(np, pd):
    def split_known_neighborhood_buildings(
        modeling_table: pd.DataFrame,
        *,
        test_fraction: float,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split buildings reproducibly while retaining neighborhood context."""
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be strictly between zero and one")
        required_columns = {"building_id", "neighborhood_id"}
        missing_columns = required_columns.difference(modeling_table.columns)
        if missing_columns:
            raise ValueError(f"Missing split columns: {sorted(missing_columns)}")
        if not modeling_table["building_id"].is_unique:
            raise ValueError("building_id must be unique before splitting")

        rng = np.random.default_rng(seed)
        test_building_ids: list[object] = []
        for _, neighborhood_df in modeling_table.groupby("neighborhood_id", sort=True):
            building_ids = np.sort(neighborhood_df["building_id"].to_numpy())
            building_count = len(building_ids)
            if building_count == 1:
                continue
            test_count = max(
                1,
                min(round(building_count * test_fraction), building_count - 1),
            )
            shuffled_ids = rng.permutation(building_ids)
            test_building_ids.extend(shuffled_ids[:test_count].tolist())

        test_mask = modeling_table["building_id"].isin(test_building_ids)
        train_df = modeling_table.loc[~test_mask].copy()
        test_df = modeling_table.loc[test_mask].copy()
        return train_df, test_df

    return (split_known_neighborhood_buildings,)


@app.cell
def _(modeling_config, modeling_df, split_known_neighborhood_buildings):
    modeling_split_seed = int(modeling_config.simulation.seed)
    modeling_test_fraction = 0.2
    # Split the modeling table into training and testing sets while preserving neighborhood context
    train_df, test_df = split_known_neighborhood_buildings(
        modeling_df,
        test_fraction=modeling_test_fraction,
        seed=modeling_split_seed,
    )

    print(
        f"Train: {len(train_df):,} buildings across {train_df['neighborhood_id'].nunique()} neighborhoods"
    )
    print(
        f"Test: {len(test_df):,} buildings across {test_df['neighborhood_id'].nunique()} neighborhoods"
    )
    return test_df, train_df


@app.cell
def _(modeling_df, pd, test_df, train_df):
    neighborhood_allocation_df = (
        pd.concat(
            [
                modeling_df.groupby("neighborhood_id").size().rename("full"),
                train_df.groupby("neighborhood_id").size().rename("train"),
                test_df.groupby("neighborhood_id").size().rename("test"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    neighborhood_allocation_df.head(12)
    return


if __name__ == "__main__":
    app.run()
