"""Tests for the components: the estimator each family makes, and the read-back from a study's params."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    LGBMParams,
    LogisticParams,
    NB2Params,
    PoissonGLMParams,
    ProbabilityComponent,
)
from age_group_prediction.modeling import NegativeBinomialRegressor

PASSTHROUGH = Plan(name="passthrough")
TRANSFORMERS = {
    "ses_only": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "median_age_only": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
LGBM = LGBMParams(
    max_depth=3,
    num_leaves=7,
    min_child_samples=5,
    learning_rate=0.1,
    n_estimators=50,
    reg_alpha=1e-8,
    reg_lambda=1e-8,
    min_split_gain=0.0,
    subsample=0.8,
    colsample_bytree=0.9,
)
# LightGBM's fixed settings: subsample needs subsample_freq; one thread per fit.
LGBM_FIXED = {"subsample_freq": 1, "n_jobs": 1, "verbosity": -1, "random_state": 7}


def _count(family: Any, estimator_params: Any, **overrides: Any) -> CountComponent:
    settings = {
        "name": "n_elementary",
        "family": family,
        "use_exposure": True,
        "estimator_params": estimator_params,
        "feature_transformer": TRANSFORMERS["ses_only"],
    }
    return CountComponent(**(settings | overrides))


def _probability(family: Any, estimator_params: Any) -> ProbabilityComponent:
    return ProbabilityComponent(
        name="probability",
        family=family,
        calibration_method="sigmoid",
        estimator_params=estimator_params,
        feature_transformer=TRANSFORMERS["ses_only"],
    )


@pytest.mark.parametrize(
    ("family", "estimator_params", "estimator_class", "fixed"),
    [
        ("lgbm_poisson", LGBM, LGBMRegressor, LGBM_FIXED | {"objective": "poisson"}),
        (
            "lgbm_gaussian",
            LGBM,
            LGBMRegressor,
            LGBM_FIXED | {"objective": "regression"},
        ),
        (
            "poisson_glm",
            PoissonGLMParams(alpha=0.1),
            PoissonRegressor,
            {"max_iter": 1000},
        ),
        ("nb2", NB2Params(dispersion=0.15, alpha=0.01), NegativeBinomialRegressor, {}),
    ],
)
def test_a_count_component_makes_its_familys_estimator_with_its_params(
    family: str, estimator_params: Any, estimator_class: type, fixed: dict[str, Any]
) -> None:
    # The record's fields are the estimator's keywords: every value reaches it,
    # beside the family's fixed settings (LightGBM's loss is the family's, D6).
    estimator = _count(family, estimator_params).make_estimator(random_state=7)
    assert type(estimator) is estimator_class
    settings = estimator.get_params()
    assert settings == settings | dataclasses.asdict(estimator_params) | fixed


@pytest.mark.parametrize(
    ("family", "estimator_params", "estimator_class", "fixed"),
    [
        ("lgbm", LGBM, LGBMClassifier, LGBM_FIXED | {"objective": "multiclass"}),
        ("logistic", LogisticParams(C=2.0), LogisticRegression, {"max_iter": 1000}),
    ],
)
def test_a_probability_component_makes_its_familys_classifier_with_its_params(
    family: str, estimator_params: Any, estimator_class: type, fixed: dict[str, Any]
) -> None:
    estimator = _probability(family, estimator_params).make_estimator(random_state=7)
    assert type(estimator) is estimator_class
    settings = estimator.get_params()
    assert settings == settings | dataclasses.asdict(estimator_params) | fixed


def test_an_unknown_family_raises_when_the_estimator_is_made() -> None:
    # A dataclass does not check the Literal; the raising else does.
    with pytest.raises(ValueError, match="unknown count family 'nb3'"):
        _count("nb3", LGBM).make_estimator()
    with pytest.raises(ValueError, match="unknown probability family 'svm'"):
        _probability("svm", LGBM).make_estimator()


def test_an_unknown_family_raises_when_the_component_is_read_back() -> None:
    with pytest.raises(ValueError, match="unknown count family 'nb3'"):
        CountComponent.from_params(
            {},
            name="total",
            family="nb3",  # type: ignore[arg-type]
            use_exposure=True,
            feature_transformer_dict=TRANSFORMERS,
        )
    with pytest.raises(ValueError, match="unknown probability family 'svm'"):
        ProbabilityComponent.from_params(
            {},
            name="probability",
            family="svm",  # type: ignore[arg-type]
            feature_transformer_dict=TRANSFORMERS,
        )


def test_a_component_reads_only_its_own_entries_from_the_params() -> None:
    # The cohorts' entries share their parameter names; without the filter by
    # name they would collide. A cohort name holds `_`, the split is at `__`.
    best_params = {
        "n_elementary__alpha": 0.1,
        "n_elementary__feature_transformer": "ses_only",
        "n_high_school__alpha": 9.0,
        "n_high_school__feature_transformer": "median_age_only",
    }
    component = CountComponent.from_params(
        best_params,
        name="n_elementary",
        family="poisson_glm",
        use_exposure=False,
        feature_transformer_dict=TRANSFORMERS,
    )
    assert component == _count(
        "poisson_glm", PoissonGLMParams(alpha=0.1), use_exposure=False
    )
    assert component.feature_transformer is TRANSFORMERS["ses_only"]


def test_a_shared_choice_is_read_as_every_components_own() -> None:
    # The independent cohorts model draws a shared choice once, under `shared__`.
    best_params = {
        "shared__dispersion": 0.15,
        "shared__alpha": 0.01,
        "n_elementary__feature_transformer": "ses_only",
        "n_high_school__feature_transformer": "median_age_only",
    }
    components = [
        CountComponent.from_params(
            best_params,
            name=cohort,
            family="nb2",
            use_exposure=True,
            feature_transformer_dict=TRANSFORMERS,
        )
        for cohort in ("n_elementary", "n_high_school")
    ]
    assert {component.estimator_params for component in components} == {
        NB2Params(dispersion=0.15, alpha=0.01)
    }
    assert [component.feature_transformer for component in components] == [
        TRANSFORMERS["ses_only"],
        TRANSFORMERS["median_age_only"],
    ]


def test_the_probability_component_reads_its_calibration_and_not_the_totals_params() -> (
    None
):
    # The total and the probabilities share LightGBM's parameter names.
    lgbm_entries = dataclasses.asdict(LGBM)
    best_params = {f"total__{name}": value for name, value in lgbm_entries.items()}
    best_params["total__max_depth"] = 8
    best_params |= {
        f"probability__{name}": value for name, value in lgbm_entries.items()
    }
    best_params |= {
        "probability__calibration_method": None,
        "probability__feature_transformer": "median_age_only",
    }
    component = ProbabilityComponent.from_params(
        best_params,
        name="probability",
        family="lgbm",
        feature_transformer_dict=TRANSFORMERS,
    )
    assert component.estimator_params == LGBM
    assert component.calibration_method is None
    assert component.feature_transformer is TRANSFORMERS["median_age_only"]


@pytest.mark.parametrize(
    ("best_params", "missing"),
    [
        ({"total__feature_transformer": "ses_only"}, "dispersion"),
        ({"total__dispersion": 0.15, "total__alpha": 0.01}, "feature_transformer"),
    ],
)
def test_a_missing_entry_in_the_params_is_named(
    best_params: dict[str, Any], missing: str
) -> None:
    # The read-back takes every entry it needs by name, so params without one
    # fail naming it, rather than building a component with a default.
    with pytest.raises(KeyError, match=missing):
        CountComponent.from_params(
            best_params,
            name="total",
            family="nb2",
            use_exposure=True,
            feature_transformer_dict=TRANSFORMERS,
        )


def test_a_missing_calibration_in_the_params_is_named() -> None:
    # `None` is a calibration choice, so a missing entry must not read as one.
    best_params = {
        "probability__C": 1.0,
        "probability__feature_transformer": "ses_only",
    }
    with pytest.raises(KeyError, match="calibration_method"):
        ProbabilityComponent.from_params(
            best_params,
            name="probability",
            family="logistic",
            feature_transformer_dict=TRANSFORMERS,
        )


def test_the_record_of_a_component_is_plain_data_with_the_transformers_declaration() -> (
    None
):
    record = json.loads(json.dumps(_probability("lgbm", LGBM).to_dict()))
    assert record["estimator_params"] == dataclasses.asdict(LGBM)
    assert record["calibration_method"] == "sigmoid"
    assert record["feature_transformer"] == json.loads(
        json.dumps(TRANSFORMERS["ses_only"].to_dict())
    )
