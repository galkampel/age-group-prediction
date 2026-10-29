"""Tests for the search spaces: the trial's names, the candidates, the read-back of the draws."""

from __future__ import annotations

import dataclasses

import optuna
import pytest
from optuna.samplers import RandomSampler
from optuna.trial import Trial

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    CountSearchSpace,
    ProbabilityComponent,
    ProbabilitySearchSpace,
)

PASSTHROUGH = Plan(name="passthrough")
TRANSFORMERS = {
    "ses_only": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "median_age_only": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
NAMES = tuple(TRANSFORMERS)
LGBM_NAMES = {
    "max_depth",
    "num_leaves",
    "min_child_samples",
    "learning_rate",
    "n_estimators",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
    "subsample",
    "colsample_bytree",
}


def _trial(seed: int = 0) -> Trial:
    return optuna.create_study(sampler=RandomSampler(seed=seed)).ask()


@pytest.mark.parametrize(
    ("family", "names"),
    [
        ("lgbm_poisson", LGBM_NAMES),
        ("lgbm_gaussian", LGBM_NAMES),
        ("poisson_glm", {"alpha"}),
        ("nb2", {"dispersion", "alpha"}),
    ],
)
def test_a_count_search_space_draws_its_familys_hyperparameters_under_its_name(
    family: str, names: set[str]
) -> None:
    trial = _trial()
    space = CountSearchSpace(
        name="n_elementary",
        family=family,  # type: ignore[arg-type]
        feature_transformer_names=NAMES,
    )
    estimator_params = space.suggest_hyperparameters(trial)
    assert set(trial.params) == {f"n_elementary__{name}" for name in names}
    assert dataclasses.asdict(estimator_params) == {
        name: trial.params[f"n_elementary__{name}"] for name in names
    }


@pytest.mark.parametrize(
    ("family", "names"), [("lgbm", LGBM_NAMES), ("logistic", {"C"})]
)
def test_a_probability_search_space_draws_its_familys_hyperparameters_under_its_name(
    family: str, names: set[str]
) -> None:
    trial = _trial()
    space = ProbabilitySearchSpace(
        name="probability",
        family=family,  # type: ignore[arg-type]
        feature_transformer_names=NAMES,
    )
    estimator_params = space.suggest_hyperparameters(trial)
    assert set(trial.params) == {f"probability__{name}" for name in names}
    assert dataclasses.asdict(estimator_params) == {
        name: trial.params[f"probability__{name}"] for name in names
    }


def test_an_unknown_family_raises_when_its_hyperparameters_are_drawn() -> None:
    with pytest.raises(ValueError, match="unknown count family 'nb3'"):
        CountSearchSpace(
            name="total",
            family="nb3",  # type: ignore[arg-type]
            feature_transformer_names=NAMES,
        ).suggest_hyperparameters(_trial())
    with pytest.raises(ValueError, match="unknown probability family 'svm'"):
        ProbabilitySearchSpace(
            name="probability",
            family="svm",  # type: ignore[arg-type]
            feature_transformer_names=NAMES,
        ).suggest_hyperparameters(_trial())


def test_the_transformer_is_drawn_by_name_from_the_injected_candidates() -> None:
    # The candidates are the notebook's data, in the order a trial draws them;
    # Optuna's choices must be primitives, so the name is drawn, not the object.
    trial = _trial()
    space = CountSearchSpace(
        name="total", family="poisson_glm", feature_transformer_names=NAMES
    )
    name = space.suggest_feature_transformer_name(trial)
    distribution = trial.distributions["total__feature_transformer"]
    assert isinstance(distribution, optuna.distributions.CategoricalDistribution)
    assert distribution.choices == NAMES
    assert name in NAMES


def test_the_calibration_is_drawn_from_the_search_spaces_candidates() -> None:
    trial = _trial()
    space = ProbabilitySearchSpace(
        name="probability", family="logistic", feature_transformer_names=NAMES
    )
    method = space.suggest_calibration_method(trial)
    distribution = trial.distributions["probability__calibration_method"]
    assert isinstance(distribution, optuna.distributions.CategoricalDistribution)
    assert distribution.choices == (None, "temperature", "sigmoid")
    assert method in ProbabilitySearchSpace.CALIBRATION_METHODS


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("family", ["lgbm_poisson", "nb2"])
@pytest.mark.parametrize("share_hyperparams", [False, True])
def test_a_count_component_read_back_from_the_trials_params_is_the_one_drawn(
    seed: int, family: str, share_hyperparams: bool
) -> None:
    # The refit reads the best components from best_params (D25.4): from a
    # trial's params, the component equals the one built from its draws,
    # whether the hyperparameters are the cohort's own or drawn under `shared__`.
    trial = _trial(seed)
    cohort_space = CountSearchSpace(
        name="n_elementary",
        family=family,  # type: ignore[arg-type]
        feature_transformer_names=NAMES,
    )
    hyperparameter_space = (
        dataclasses.replace(cohort_space, name="shared")
        if share_hyperparams
        else cohort_space
    )
    drawn = CountComponent(
        name="n_elementary",
        family=family,  # type: ignore[arg-type]
        use_exposure=True,
        estimator_params=hyperparameter_space.suggest_hyperparameters(trial),
        feature_transformer=TRANSFORMERS[
            cohort_space.suggest_feature_transformer_name(trial)
        ],
    )
    read = CountComponent.from_params(
        trial.params,
        name="n_elementary",
        family=family,  # type: ignore[arg-type]
        use_exposure=True,
        feature_transformer_dict=TRANSFORMERS,
    )
    assert read == drawn


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("family", ["lgbm", "logistic"])
def test_a_probability_component_read_back_from_the_trials_params_is_the_one_drawn(
    seed: int, family: str
) -> None:
    trial = _trial(seed)
    space = ProbabilitySearchSpace(
        name="probability",
        family=family,  # type: ignore[arg-type]
        feature_transformer_names=NAMES,
    )
    drawn = ProbabilityComponent(
        name="probability",
        family=family,  # type: ignore[arg-type]
        estimator_params=space.suggest_hyperparameters(trial),
        calibration_method=space.suggest_calibration_method(trial),
        feature_transformer=TRANSFORMERS[space.suggest_feature_transformer_name(trial)],
    )
    read = ProbabilityComponent.from_params(
        trial.params,
        name="probability",
        family=family,  # type: ignore[arg-type]
        feature_transformer_dict=TRANSFORMERS,
    )
    assert read == drawn
