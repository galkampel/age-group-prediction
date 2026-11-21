"""C4: the check store against docs/MLFLOW_RUNS.md (§1, §4-§8) after C2-C3c, read-only.

Usage: python checks/check_store.py <db copy> <experiment> <stage...>   (stages: empty, tuning, best_runs, final)
Run it on a COPY of the store; set MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false to silence the progress bars.
The caller copies the live sqlite file first; the artifacts are read from their absolute URIs.
Every failed expectation is printed as FAIL; the exit code is the number of failures.
"""

import collections
import io
import json
import subprocess
import sys

import mlflow
import pandas as pd

db, experiment_name, *stages = sys.argv[1:]
client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{db}")
REPO = "/Users/galkampel/Desktop/Projects/age-group-prediction"
HEAD = subprocess.run(
    ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
MODELS = ("independent_cohorts", "total_times_probability")
METRICS = 6
COHORTS = ("n_kindergarten", "n_elementary", "n_high_school")  # replaced below by what the store holds
STATISTICS = {"score", "ci95_lower", "ci95_upper"}
SETTINGS = {
    "config_hash", "config_path", "n_neighborhoods", "buildings_per_neighborhood_rate",
    "simulation_seed", "seed", "split_method", "test_size", "val_size", "n_splits",
    "n_trials", "n_startup_trials", "objective_metric", "selection_metric", "n_resamples",
    "alpha", "n_repeats", "n_shap_rows", "n_shap_background", "n_train", "n_val", "n_test",
    "prediction_floor",
}
KEPT = {"config_hash", "seed", "split_method", "n_trials", "objective_metric"}
RUN_SETTINGS = {
    "independent_cohorts": {"family", "use_exposure", "share_hyperparams", "share_feature_transformer"},
    "total_times_probability": {"total_family", "probability_family", "use_exposure"},
}
CODE_TAGS = {"run_key", "study_folder", "stage", "notebook", "git_commit", "git_dirty"}
TUNING_ARTIFACTS = {
    "best_params.json", "components.json", "config.json", "search_space.json", "settings.json", "scores.csv",
    "trials.csv", "figures/optimization_history.html", "figures/slice.html",
}
failures = []


def expect(condition, message):
    if not condition:
        failures.append(message)
        print("FAIL", message)


def walk(run_id, path=None):
    out = []
    for item in client.list_artifacts(run_id, path):
        out += walk(run_id, item.path) if item.is_dir else [item.path]
    return out


def own_tags(run):
    return {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}


def inputs(run):
    return sorted(
        (tag.value, i.dataset.name, len(i.dataset.digest))
        for i in run.inputs.dataset_inputs
        for tag in i.tags
        if tag.key == "mlflow.data.context"
    )


def settings_seed(run):
    return mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/settings.json")["study_seed"]


def check_common(run):
    tags = own_tags(run)
    expect(CODE_TAGS <= set(tags), f"{run.info.run_name}: code tags {sorted(tags)}")
    expect(tags.get("notebook") == "notebooks/03_hyperparameter_tuning.py", f"{run.info.run_name}: notebook tag")
    expect(tags.get("git_commit") == HEAD, f"{run.info.run_name}: git_commit {tags.get('git_commit')} != HEAD {HEAD}")
    if tags.get("git_dirty") != "false":  # a note, not a failure: C4's own edits are uncommitted while it checks
        print("NOTE", f"{run.info.run_name}: git_dirty {tags.get('git_dirty')}")
    expect(KEPT <= set(run.data.params), f"{run.info.run_name}: missing kept params {sorted(KEPT - set(run.data.params))}")
    expect(run.info.status == "FINISHED", f"{run.info.run_name}: status {run.info.status}")
    expect(tags.get("study_folder") == tags.get("run_key", "").split("/")[0], f"{run.info.run_name}: study_folder tag")
    settings = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/settings.json")
    extra = {"tuning": {"study_seed"}, "final": {"n_train_val"}}.get(tags.get("stage"), set())
    expect(set(settings) == SETTINGS | extra, f"{run.info.run_name}: settings.json {sorted(set(settings) ^ (SETTINGS | extra))}")
    expect(all(str(settings[name]) == run.data.params[name] for name in KEPT if name in run.data.params), f"{run.info.run_name}: params differ from settings.json")


experiment = client.get_experiment_by_name(experiment_name)
expect(experiment is not None, f"experiment {experiment_name} missing")
print("experiment", experiment.name, "| id", experiment.experiment_id, "| artifact location", experiment.artifact_location)
runs = client.search_runs([experiment.experiment_id], max_results=1000, run_view_type=mlflow.entities.ViewType.ALL)
by_stage = collections.defaultdict(list)
for run in runs:
    by_stage[(run.data.tags.get("stage"), run.info.status, run.info.lifecycle_stage)].append(run)
print("runs by (stage, status, lifecycle):", {k: len(v) for k, v in by_stage.items()})
finished = lambda stage: by_stage[(stage, "FINISHED", "active")]  # noqa: E731

if "empty" in stages:
    expect(len(runs) == 0, f"{len(runs)} runs in a new store")

if "tuning" in stages:
    tuning = finished("tuning")
    expect(len(tuning) == 24, f"{len(tuning)} tuning runs")
    keys = [r.data.tags["run_key"] for r in tuning]
    expect(len(set(keys)) == 24, "tuning run_keys not distinct")
    folders = {k.split("/")[0] for k in keys}
    print("study folder(s):", folders)
    expect(len(folders) == 1, "one study folder")
    print("models:", collections.Counter(r.data.tags["model"] for r in tuning))
    expect(collections.Counter(r.data.tags["model"] for r in tuning) == {MODELS[0]: 8, MODELS[1]: 16}, "8 + 16 runs")
    first = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{tuning[0].info.artifact_uri}/scores.csv")))
    metric_names = sorted(first["metric"].unique())
    COHORTS = tuple(c for c in first["cohort"].unique() if c != "composite")
    expect(len(metric_names) == METRICS and len(COHORTS) == 3, f"scores.csv: {len(metric_names)} metrics, cohorts {COHORTS}")
    for run in tuning:
        check_common(run)
        name = run.info.run_name
        study_name = run.data.tags["run_key"].split("/")[1]
        objective = run.data.params["objective_metric"]
        expected_metrics = (
            {f"cv/{objective}", "seconds", "n_complete", "n_failed"}
            | {f"{part}/composite/{metric}" for part in ("train", "val") for metric in metric_names}
            | {f"val/{cohort}/{objective}" for cohort in COHORTS}
        )
        expect(set(run.data.metrics) == expected_metrics and len(run.data.metrics) == 19, f"{name}: {len(run.data.metrics)} metrics, diff {sorted(set(run.data.metrics) ^ expected_metrics)}")
        model = run.data.tags["model"]
        expect(set(run.data.params) - {p for p in run.data.params if p.startswith("best.")} == RUN_SETTINGS[model] | KEPT | {"study_seed"}, f"{name}: params {sorted(run.data.params)}")
        expect(len(run.data.params) - sum(p.startswith("best.") for p in run.data.params) == (10 if model == MODELS[0] else 9), f"{name}: param count")
        expect(run.data.metrics["n_complete"] == int(run.data.params["n_trials"]) and run.data.metrics["n_failed"] == 0, f"{name}: n_complete {run.data.metrics['n_complete']} n_failed {run.data.metrics['n_failed']}")
        paths = sorted(walk(run.info.run_id))
        trials = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/trials.csv")))
        ranks = trials.loc[trials["state"] == "COMPLETE", "value"].nunique() > 1  # fANOVA ranks only then
        wanted = TUNING_ARTIFACTS | {f"study/{study_name}.log"} | ({"figures/param_importances.html"} if ranks else set())
        expect(paths == sorted(wanted), f"{name}: artifacts {sorted(set(paths) ^ wanted)}")
        scores = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/scores.csv")))
        expect(list(scores.columns) == ["dataset", "cohort", "metric", "score"] and len(scores) == 48, f"{name}: scores.csv {scores.shape}")
        expect(settings_seed(run) == int(run.data.params["study_seed"]), f"{name}: study_seed")
        expect(inputs(run) == [("training", "train", 36), ("validation", "val", 36)], f"{name}: inputs {inputs(run)}")
        best_params = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/best_params.json")
        expect({f"best.{k}" for k in best_params} == {p for p in run.data.params if p.startswith("best.")}, f"{name}: best.* params")
        expect(run.data.tags["family"] in name, f"{name}: family tag {run.data.tags['family']}")
    for model in MODELS:
        run = next(r for r in tuning if r.data.tags["model"] == model)
        trials = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/trials.csv")))
        print(f"  {model}: e.g. {run.info.run_name!r}")
        print("    params:", sorted(p for p in run.data.params if not p.startswith("best.")), "+", sum(p.startswith("best.") for p in run.data.params), "best.*")
        print("    trials.csv", trials.shape, "states", dict(trials["state"].value_counts()))
        print("    metric groups", dict(collections.Counter("/".join(k.split("/")[:2]) for k in run.data.metrics)))

if "best_runs" in stages or "final" in stages:
    tuning_keys = {r.data.tags["run_key"] for r in finished("tuning")}
    for stage, part, n_tables, n_figures, contexts in (
        ("best_runs", "val", 7, 18, [("training", "train", 36), ("validation", "val", 36)]),
        ("final", "test", 7, 33, [("testing", "test", 36), ("training", "train_val", 36)]),  # W2: §9's table and 10 waterfalls
    ):
        if stage not in stages:
            continue
        found = finished(stage)
        expect(len(found) == 1, f"{len(found)} FINISHED {stage} runs")
        (run,) = found
        check_common(run)
        tags = own_tags(run)
        print(f"---- {run.info.run_name}: tags", {k: v for k, v in tags.items() if k not in {"git_commit"}})
        folder = run.data.tags["run_key"].split("/")[0]
        expected_key = f"{folder}/final" if stage == "final" else f"{folder}/best_runs__objective={run.data.params['objective_metric']}__selection={run.data.params['selection_metric']}"
        expect(tags["run_key"] == expected_key, f"{stage}: run_key {tags['run_key']}")
        expect(stage != "final" or tags.get("final") == "true", "final=true tag")
        for model in MODELS:
            expect(f"{folder}/{tags[f'best.{model}']}" in tuning_keys, f"{stage}: best.{model} names no tuning run")
        expect(set(run.data.params) == KEPT | {"selection_metric"}, f"{stage}: params {sorted(run.data.params)}")
        expect(tags["study_folder"] == folder, f"{stage}: study_folder tag")
        by_key = {r.data.tags["run_key"]: r for r in finished("tuning")}
        for model in MODELS:
            linked = client.get_run(tags[f"best.{model}.run_id"])
            expect(linked.info.status == "FINISHED" and linked.data.tags["stage"] == "tuning", f"{stage}: best.{model}.run_id not a FINISHED tuning run")
            expect(linked.data.tags["model"] == model and linked.data.tags["run_key"] == f"{folder}/{tags[f'best.{model}']}", f"{stage}: best.{model}.run_id names {linked.data.tags['run_key']}")
            expect(by_key.get(f"{folder}/{tags[f'best.{model}']}") is not None and by_key[f"{folder}/{tags[f'best.{model}']}"].info.run_id == linked.info.run_id, f"{stage}: best.{model}.run_id differs from the run_key search")
        metrics = run.data.metrics
        expect(len(metrics) == 54, f"{stage}: {len(metrics)} metrics")
        parts = {k.split("/") [0] for k in metrics}
        expect(parts == {part}, f"{stage}: metric prefixes {parts}")
        expect({k.split("/")[1] for k in metrics} == {*MODELS, "difference"}, f"{stage}: models {sorted({k.split('/')[1] for k in metrics})}")
        expect({k.split("/")[3] for k in metrics} == STATISTICS, f"{stage}: statistics")
        expect(len({k.split("/")[2] for k in metrics}) == METRICS, f"{stage}: metrics")
        paths = walk(run.info.run_id)
        tables = sorted(p for p in paths if p.startswith("tables/"))
        figures = sorted(p for p in paths if p.startswith("figures/"))
        others = sorted(p for p in paths if not p.startswith(("tables/", "figures/")) and p != "settings.json")
        expect("settings.json" in paths, f"{stage}: no settings.json")
        expect(len(tables) == n_tables, f"{stage}: {len(tables)} tables")
        expect(len(figures) == n_figures, f"{stage}: {len(figures)} figures")
        if stage == "final":  # W2 (MLFLOW_RUNS_PLAN.md §9): the SHAP values table and each model's 5 worst waterfalls
            expect("tables/test_shap_values.csv" in tables, "final: no tables/test_shap_values.csv")
            for model in MODELS:
                waterfalls = [f for f in figures if f.startswith(f"figures/test_waterfall_{model}_total_")]
                ranks = sorted(int(f.removeprefix(f"figures/test_waterfall_{model}_total_").split("_")[0]) for f in waterfalls)
                expect(ranks == [1, 2, 3, 4, 5], f"final: {model}'s waterfalls {waterfalls}")
                print(f"    {model}'s waterfalls:", [f.removeprefix(f"figures/test_waterfall_{model}_total_").removesuffix(".html") for f in sorted(waterfalls)])
            shap_values = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/tables/test_shap_values.csv")))
            expect(list(shap_values.columns[:4]) == ["model", "building_id", "cohort", "base_value"], f"final: test_shap_values columns {list(shap_values.columns)}")
            expect(len(shap_values) == 2 * 3 * int(mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/settings.json")["n_test"]), f"final: test_shap_values {len(shap_values)} rows")
            print("    test_shap_values", shap_values.shape)
        expect(others == sorted(f"{kind}/{model}.json" for kind in ("best_params", "components") for model in MODELS), f"{stage}: others {others}")
        expect(inputs(run) == contexts, f"{stage}: inputs {inputs(run)}")
        print("    tables", [t.removeprefix("tables/") for t in tables])
        print("    figures", len(figures), "| others", others)
        for model in MODELS:
            components = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/components/{model}.json")
            print(f"    best.{model} = {tags[f'best.{model}']}")
            print("      components", {name: {k: v for k, v in c.items() if k in ("family", "use_exposure")} for name, c in components.items()})
        if stage == "best_runs":
            selection = run.data.params["selection_metric"]
            for model in MODELS:
                ranked = sorted((r for r in finished("tuning") if r.data.tags["model"] == model), key=lambda r: r.data.metrics[f"val/composite/{selection}"])
                expect(ranked[0].data.tags["run_key"] == f"{folder}/{tags[f'best.{model}']}", f"§7: {model}'s best by {selection} is not best.{model}")
        if stage == "final":
            scores = {k: v for k, v in metrics.items() if k.endswith("/score")}
            print("    test scores:")
            for key in sorted(scores):
                print(f"      {key} = {scores[key]!r}")
            print("    test_scores.csv:")
            print(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/tables/test_scores.csv"))

for run in runs:
    if run.info.status != "FINISHED":
        expect(not run.data.metrics, f"{run.info.run_name}: {run.info.status} run holds {len(run.data.metrics)} metrics")
print("FAILURES:", len(failures))
sys.exit(len(failures))
