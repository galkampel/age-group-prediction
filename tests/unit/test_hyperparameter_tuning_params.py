"""Tests for the params records: keywords only, frozen, a typo caught when built."""

from __future__ import annotations

import dataclasses

import pytest

from age_group_prediction.hyperparameter_tuning import (
    LGBMParams,
    LogisticParams,
    NB2Params,
    PoissonGLMParams,
)

LGBM_VALUES = {
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 5,
    "learning_rate": 0.1,
    "n_estimators": 50,
    "reg_alpha": 1e-8,
    "reg_lambda": 1e-8,
    "min_split_gain": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


def test_a_typo_in_a_hyperparameter_name_is_caught_when_the_record_is_built() -> None:
    # LightGBM accepts and fits `max_dept=3` silently (NOTEBOOK_TUNING_PLAN.md
    # D11), so the record is where the name is checked.
    values = dict(LGBM_VALUES)
    values["max_dept"] = values.pop("max_depth")
    with pytest.raises(TypeError, match="max_dept"):
        LGBMParams(**values)


@pytest.mark.parametrize(
    ("params_class", "values"),
    [
        (LGBMParams, tuple(LGBM_VALUES.values())),
        (PoissonGLMParams, (0.1,)),
        (NB2Params, (0.15, 0.01)),
        (LogisticParams, (1.0,)),
    ],
)
def test_a_record_built_by_position_is_refused(
    params_class: type, values: tuple[float, ...]
) -> None:
    # Keywords only: a value cannot land on the wrong hyperparameter.
    with pytest.raises(TypeError):
        params_class(*values)


def test_a_record_cannot_be_changed_after_it_is_built() -> None:
    params = NB2Params(dispersion=0.15, alpha=0.01)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.alpha = 0.1  # type: ignore[misc]
