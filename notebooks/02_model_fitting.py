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
    # Building-Level Age-Group Modeling: Gate 8 Final Client

    This is a thin client over `age_group_prediction`. It never fits, predicts,
    or evaluates a model directly; every statistical step goes through a package
    API that Gates 1-8 already built and tested. The notebook's own job is:

    1. Build the canonical population and modeling table.
    2. Create or load the persisted outer test-lockbox manifest, without ever
       displaying a holdout target.
    3. Build the canonical candidate registry and training-only validation
       folds.
    4. Run and track cross-validation (Gate 6/7) behind an explicit button.
    5. Display within-approach selections and CV evidence.
    6. Compute and display the cross-family decision (Gate 8), from training/CV
       evidence only.
    7. Run the linked final refit, one-time lockbox evaluation, and MLflow
       logging (Gate 8) behind a second button plus a confirmation checkbox.
    8. Display final metrics, loadable model URIs, and MLflow run links.

    A plain script run (`uv run notebooks/02_model_fitting.py`) performs setup
    and validation only. Setup loads (or first creates) the persisted manifest
    and replays it in memory to obtain the training partition, but no holdout
    row is displayed or evaluated. Both run buttons default to unclicked, so no
    cross-validation starts, no full Bayesian model fits, and no MLflow run is
    created.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Setup: Imports And Configuration
    """)
    return


@app.cell
def _():
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from age_group_prediction import (
        DEFAULT_MODELING_SCHEMA,
        ExperimentConfig,
        build_modeling_table,
        load_experiment_config,
        load_split_manifest,
        make_validation_folds,
        persist_split_manifest,
        replay_split_manifest,
        split_known_neighborhood_buildings,
    )
    from age_group_prediction.experiment import (
        build_canonical_candidate_registry,
        run_cross_model_validation,
        select_cross_family_winner,
    )
    from age_group_prediction.tracking import (
        TrackingContext,
        log_cross_validation_experiment,
        resolve_tracking_settings,
        run_final_evaluation,
    )
    from student_simulator import StudentPopulationSimulator, load_simulation_config

    return (
        DEFAULT_MODELING_SCHEMA,
        ExperimentConfig,
        Path,
        StudentPopulationSimulator,
        TrackingContext,
        build_canonical_candidate_registry,
        build_modeling_table,
        load_experiment_config,
        load_simulation_config,
        load_split_manifest,
        log_cross_validation_experiment,
        make_validation_folds,
        np,
        pd,
        persist_split_manifest,
        replace,
        replay_split_manifest,
        resolve_tracking_settings,
        run_cross_model_validation,
        run_final_evaluation,
        select_cross_family_winner,
        split_known_neighborhood_buildings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Canonical Population And Modeling Table

    Reproduces the same population the EDA notebook analyzed: `150`
    neighborhoods at a `9.0` buildings-per-neighborhood rate. `build_modeling_table`
    is the single package entry point that selects raw columns, derives the
    non-collinear room-share representation, orders columns, and validates the
    schema invariants -- the same construction every model in this project
    trains against.
    """)
    return


@app.cell
def _(StudentPopulationSimulator, load_simulation_config, mo):
    project_root = mo.notebook_dir().parent
    simulation_config_path = project_root / "configs" / "simulation.toml"
    base_simulation_config = load_simulation_config(simulation_config_path)
    canonical_simulation_config = base_simulation_config.model_copy(
        update={
            "simulation": base_simulation_config.simulation.model_copy(
                update={"n_neighborhoods": 150}
            ),
            "building": base_simulation_config.building.model_copy(
                update={"buildings_per_neighborhood_rate": 9.0}
            ),
        }
    )
    population_df = StudentPopulationSimulator(canonical_simulation_config).run()
    mo.md(
        f"""
        **Canonical population:** `{len(population_df):,}` buildings across
        `{population_df["neighborhood_id"].nunique()}` neighborhoods.

        - Configuration: `{simulation_config_path.relative_to(project_root)}`
        - Simulation seed: `{canonical_simulation_config.simulation.seed}`
        """
    )
    return canonical_simulation_config, population_df, project_root


@app.cell
def _(DEFAULT_MODELING_SCHEMA, build_modeling_table, mo, population_df):
    modeling_df = build_modeling_table(population_df)
    # No split exists yet at this point, so every row here could still become
    # a holdout building; target columns are dropped from the preview so this
    # setup-only cell never discloses a target before the lockbox is even
    # created (section 4).
    preview_columns = [
        column
        for column in modeling_df.columns
        if column not in DEFAULT_MODELING_SCHEMA.target_columns
    ]
    mo.vstack(
        [
            mo.md(
                f"**Modeling table:** `{len(modeling_df):,}` rows, schema-validated. "
                "Target columns are withheld from this preview: every row here may "
                "still become a holdout building once the manifest below is created."
            ),
            modeling_df.loc[:, preview_columns].head(5),
        ]
    )
    return (modeling_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Experiment Configuration

    Loads the runtime defaults (splits, folds, tuning, likelihood, Bayesian
    profiles) that every candidate the registry declares is built from.
    """)
    return


@app.cell
def _(load_experiment_config, mo, project_root):
    experiment_config_path = project_root / "configs" / "modeling.toml"
    experiment_config = load_experiment_config(experiment_config_path)
    mo.md(
        f"**Experiment configuration:** `{experiment_config_path.relative_to(project_root)}`"
    )
    return experiment_config, experiment_config_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Persisted Outer Test-Lockbox Manifest

    Created once and replayed on every later run: `persist_split_manifest`
    refuses to silently replace a manifest already on disk. Only building
    counts are shown here, never a holdout row or target.
    """)
    return


@app.cell
def _(
    experiment_config,
    load_split_manifest,
    modeling_df,
    np,
    persist_split_manifest,
    project_root,
    replay_split_manifest,
    split_known_neighborhood_buildings,
):
    lockbox_manifest_path = project_root / "artifacts" / "lockbox" / "split_manifest.json"
    outer_split_config = experiment_config.outer_split

    if lockbox_manifest_path.exists():
        split_manifest = load_split_manifest(lockbox_manifest_path)
        manifest_origin = "loaded from disk"
    else:
        fresh_split = split_known_neighborhood_buildings(
            modeling_df,
            config=outer_split_config,
            rng=np.random.default_rng(experiment_config.randomness.default_seed),
        )
        split_manifest = persist_split_manifest(lockbox_manifest_path, fresh_split.manifest)
        manifest_origin = "created and persisted"

    split = replay_split_manifest(modeling_df, split_manifest, config=outer_split_config)
    return lockbox_manifest_path, manifest_origin, outer_split_config, split, split_manifest


@app.cell
def _(lockbox_manifest_path, manifest_origin, mo, project_root, split):
    mo.vstack(
        [
            mo.md(
                f"""
                **Outer manifest** ({manifest_origin}):
                `{lockbox_manifest_path.relative_to(project_root)}`

                - Training buildings: `{len(split.manifest.training_building_ids):,}`
                - Holdout buildings: `{len(split.manifest.holdout_building_ids):,}`
                  (rows and targets not displayed)
                - Realized holdout fraction: `{split.manifest.realized_holdout_fraction:.3f}`
                """
            ),
            mo.md("### Training-only summary (holdout excluded)"),
            split.train_df.describe(percentiles=[0.1, 0.5, 0.9]).round(3).T,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Canonical Candidates And Validation Folds

    `build_canonical_candidate_registry` is the single production source for
    candidates, selection policies, and full-training refit factories --
    exactly what Gate 6/7 already validated, never a notebook-local
    reconstruction.
    """)
    return


@app.cell
def _(build_canonical_candidate_registry, experiment_config, make_validation_folds, mo, np, split):
    candidate_registry = build_canonical_candidate_registry(experiment_config)
    validation_fold_plan = make_validation_folds(
        split.train_df,
        config=experiment_config.folds,
        rng=np.random.default_rng(experiment_config.randomness.default_seed),
    )
    mo.md(
        f"""
        **Candidate set:** `{candidate_registry.candidate_set_name}`
        (`{len(candidate_registry.candidates)}` candidates,
        fingerprint `{candidate_registry.candidate_set_fingerprint[:12]}...`)

        **Validation folds:** `{len(validation_fold_plan.folds)}`
        (`{len(validation_fold_plan.unvalidated_building_ids)}` singleton-neighborhood
        buildings train-only, never validated)
        """
    )
    return candidate_registry, validation_fold_plan


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Run And Track Cross-Validation (Gate 6/7)

    Explicit action: nothing above this point fits a model. Capturing
    artifacts (state bundles with reload checks) is required for tracking.
    """)
    return


@app.cell
def _(mo):
    run_cv_button = mo.ui.run_button(label="Run cross-validation")
    run_cv_button
    return (run_cv_button,)


@app.cell
def _(
    TrackingContext,
    candidate_registry,
    experiment_config,
    log_cross_validation_experiment,
    mo,
    run_cross_model_validation,
    run_cv_button,
    split,
    validation_fold_plan,
):
    mo.stop(
        not run_cv_button.value,
        mo.md("Click **Run cross-validation** above to start Gate 6/7."),
    )
    cv_result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=validation_fold_plan.folds,
        candidates=candidate_registry.candidates,
        selection_policies=candidate_registry.selection_policies,
        experiment_config=experiment_config,
        capture_artifacts=True,
    )
    source_cv_run_id = log_cross_validation_experiment(
        cv_result,
        TrackingContext(run_name="gate8-cv"),
        train_df=split.train_df,
    )
    mo.md(f"**Cross-validation complete.** MLflow parent run: `{source_cv_run_id}`")
    return cv_result, source_cv_run_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Within-Approach Selections And CV Evidence
    """)
    return


@app.cell
def _(cv_result, mo, pd):
    selection_rows = [
        {
            "approach": selection.approach,
            "selected_candidate_id": selection.selected_candidate_id,
            "rejected_candidate_ids": ", ".join(selection.rejected_candidate_ids) or "-",
        }
        for selection in cv_result.selections
    ]
    mo.vstack(
        [
            mo.md("### Frozen within-approach winners"),
            pd.DataFrame(selection_rows),
            mo.md("### Aggregate cross-validation metrics"),
            cv_result.aggregate_metrics_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8) Cross-Family Decision (Gate 8)

    `select_cross_family_winner` reads only training/CV evidence -- the three
    frozen within-approach winners and their aggregate metrics -- and records
    the decision before any holdout row is ever materialized.
    """)
    return


@app.cell
def _(cv_result, mo, select_cross_family_winner):
    cross_family_selection = select_cross_family_winner(cv_result)
    pretest_freeze = cv_result.freeze.with_cross_family_selection(cross_family_selection)
    mo.md(
        f"""
        **Cross-family winner:** `{cross_family_selection.selected_candidate_id}`
        (`{cross_family_selection.selected_approach}`)

        Decisive final criterion: `{cross_family_selection.decisive_final_criterion}`
        (tie-break used: `{cross_family_selection.final_tie_break_used}`)
        """
    )
    return cross_family_selection, pretest_freeze


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9) Run The Final Refit, Lockbox Evaluation, And Tracking (Gate 8)

    This is the **one-time** action: full-training refit of all three frozen
    winners (the Bayesian winner forced to the full NUTS profile), a single
    replay of the persisted manifest, one-time holdout evaluation, and MLflow
    logging including each refit as a loadable models-from-code pyfunc.
    `run_final_evaluation` is the single guarded package call; this notebook
    never reads the holdout partition itself and never calls a model's own
    `fit`, `predict`, or `evaluate`.
    """)
    return


@app.cell
def _(mo):
    confirm_final_checkbox = mo.ui.checkbox(
        label="I understand this opens the one-time test holdout and cannot be undone for this manifest."
    )
    run_final_button = mo.ui.run_button(label="Run final evaluation (opens lockbox)")
    mo.vstack([confirm_final_checkbox, run_final_button])
    return confirm_final_checkbox, run_final_button


@app.cell
def _(
    TrackingContext,
    candidate_registry,
    confirm_final_checkbox,
    cv_result,
    experiment_config,
    mo,
    modeling_df,
    pretest_freeze,
    run_final_button,
    run_final_evaluation,
    source_cv_run_id,
    split,
):
    mo.stop(
        not (run_final_button.value and confirm_final_checkbox.value),
        mo.md(
            "Check the confirmation box and click **Run final evaluation** "
            "above to open the lockbox."
        ),
    )
    final_evaluation_result = run_final_evaluation(
        split.train_df,
        modeling_df,
        split_manifest=split.manifest,
        cv_result=cv_result,
        pretest_freeze=pretest_freeze,
        candidates=candidate_registry.candidates,
        final_refit_factories=candidate_registry.final_refit_factories,
        context=TrackingContext(run_name="gate8-final"),
        source_cv_run_id=source_cv_run_id,
        evaluation_config=experiment_config.evaluation,
    )
    mo.md("**Final evaluation complete.**")
    return (final_evaluation_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10) Final Metrics, Model URIs, And MLflow Run Links
    """)
    return


@app.cell
def _(final_evaluation_result, mo):
    mo.vstack(
        [
            mo.md(f"**Selected:** `{final_evaluation_result.finalized_freeze.cross_family_selection.selected_candidate_id}`"),
            *[
                mo.vstack(
                    [
                        mo.md(f"### `{evaluation.candidate_id}` (`{evaluation.role}`)"),
                        evaluation.metrics_df,
                    ]
                )
                for evaluation in final_evaluation_result.evaluations
            ],
        ]
    )
    return


@app.cell
def _(final_evaluation_result, mo, pd, resolve_tracking_settings, source_cv_run_id):
    import mlflow

    tracking_settings = resolve_tracking_settings()
    mlflow_client = mlflow.MlflowClient(tracking_settings.tracking_uri)
    tracking_experiment = mlflow_client.get_experiment_by_name(
        tracking_settings.experiment_name
    )
    final_parent_runs = mlflow_client.search_runs(
        [tracking_experiment.experiment_id],
        filter_string=(
            "tags.run_role = 'final_evaluation' and "
            f"tags.source_cv_run_id = '{source_cv_run_id}'"
        ),
    )
    final_parent_run = final_parent_runs[0]
    final_child_runs = {
        run.data.tags["candidate_id"]: run
        for run in mlflow_client.search_runs(
            [tracking_experiment.experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{final_parent_run.info.run_id}'",
        )
    }
    model_uri_rows = [
        {
            "candidate_id": evaluation.candidate_id,
            "role": final_child_runs[evaluation.candidate_id].data.tags.get("role"),
            "final_model_uri": final_child_runs[evaluation.candidate_id].data.tags.get(
                "final_model_uri"
            ),
        }
        for evaluation in final_evaluation_result.evaluations
    ]
    mo.vstack(
        [
            mo.md(
                f"""
                **MLflow tracking URI:** `{tracking_settings.tracking_uri}`
                **Experiment:** `{tracking_settings.experiment_name}`
                **Source CV run:** `{source_cv_run_id}`
                **Final run:** `{final_parent_run.info.run_id}`

                Inspect with `mlflow ui --backend-store-uri {tracking_settings.tracking_uri}`.
                """
            ),
            mo.ui.table(pd.DataFrame(model_uri_rows)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
