"""Tests for the hyperparameter classes.

A trial records the distribution each ``suggest_*`` call asked for, so
comparing ``trial.distributions[name]`` with the expected distribution proves
every option reached Optuna unchanged.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import Trial

import age_group_prediction.hyperparameter_tuning as tuning
from age_group_prediction.hyperparameter_tuning import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)

NUMERIC = (FloatParameter, IntParameter)


def _trial() -> Trial:
    """A live trial from a seeded in-memory study."""
    return optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        (FloatParameter("p", 0.5, 2.0), FloatDistribution(0.5, 2.0)),
        (
            FloatParameter("p", 0.5, 2.0, log=True),
            FloatDistribution(0.5, 2.0, log=True),
        ),
        (
            FloatParameter("p", 0.5, 2.0, step=0.5),
            FloatDistribution(0.5, 2.0, step=0.5),
        ),
        (IntParameter("p", 1, 9), IntDistribution(1, 9)),
        (IntParameter("p", 1, 9, log=True), IntDistribution(1, 9, log=True)),
        (IntParameter("p", 1, 9, step=2), IntDistribution(1, 9, step=2)),
    ],
    ids=["float", "float-log", "float-step", "int", "int-log", "int-step"],
)
def test_every_option_reaches_optuna(
    parameter: FloatParameter | IntParameter, expected: BaseDistribution
) -> None:
    trial = _trial()

    value = parameter.suggest(trial)

    assert trial.distributions["p"] == expected
    assert type(value) is (float if isinstance(parameter, FloatParameter) else int)
    assert parameter.low <= value <= parameter.high


@pytest.mark.parametrize(
    "build",
    [
        lambda: FloatParameter("p", 0.0, 1.0, log=True),  # float log needs low > 0
        lambda: IntParameter("p", 0, 9, log=True),  # int log needs low >= 1
    ],
    ids=["float", "int"],
)
def test_invalid_setting_fails_at_construction(build: Callable[[], object]) -> None:
    # Optuna's own rule, raised when the parameter is built rather than at the
    # first trial, and prefixed with the parameter's name.
    with pytest.raises(ValueError, match=r"parameter 'p': .*low"):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: FloatParameter("p", 0.0, 1.0, step=0.3),  # Optuna would search [0, 0.9]
        lambda: IntParameter("p", 0, 10, step=3),  # Optuna would search [0, 9]
    ],
    ids=["float", "int"],
)
def test_step_that_does_not_divide_the_range_is_rejected(
    build: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match=r"parameter 'p': .*divisible"):
        build()


def test_step_that_divides_the_range_is_accepted() -> None:
    # Guards the escalation against false alarms from float rounding.
    FloatParameter("p", 0.1, 1.0, step=0.1)
    FloatParameter("p", 0.7, 1.0, step=0.05)


@pytest.mark.parametrize(
    "build",
    [
        lambda: FloatParameter("p", math.nan, 1.0),
        lambda: FloatParameter("p", 0.0, math.inf),
        lambda: FloatParameter("p", 0.0, 1.0, step=math.inf),
        lambda: CategoricalParameter("p", (math.nan, 1.0)),
    ],
    ids=["nan-low", "inf-high", "inf-step", "nan-choice"],
)
def test_nan_and_inf_are_rejected_at_construction(
    build: Callable[[], object],
) -> None:
    # Otherwise they pass here and fail only at the first trial, with an
    # OverflowError or decimal.InvalidOperation rather than a ValueError.
    with pytest.raises(ValueError, match="finite number"):
        build()


def test_unrelated_warnings_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only Optuna's UserWarnings are escalated; anything else reaches the caller.
    original = FloatParameter._distribution

    def noisy(self: FloatParameter) -> FloatDistribution:
        warnings.warn("some deprecation", DeprecationWarning, stacklevel=1)
        return original(self)

    monkeypatch.setattr(FloatParameter, "_distribution", noisy)
    with pytest.warns(DeprecationWarning, match="some deprecation"):
        FloatParameter("p", 0.0, 1.0)


@pytest.mark.parametrize("cls", NUMERIC)
def test_log_and_step_are_keyword_only(
    cls: type[FloatParameter | IntParameter],
) -> None:
    # Positionally, `True` after `high` would silently become `log` or `step`.
    with pytest.raises(ValueError, match="Unexpected positional argument"):
        cls("p", 1, 2, True)  # type: ignore[misc]


@pytest.mark.parametrize("cls", NUMERIC)
def test_parameter_is_frozen(cls: type[FloatParameter | IntParameter]) -> None:
    parameter = cls("p", 1, 2)
    with pytest.raises(FrozenInstanceError):
        parameter.low = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "build",
    [
        lambda: IntParameter("p", 1.5, 9),  # a float where an int belongs
        lambda: IntParameter("p", True, 9),  # bool is an int subclass in Python
        lambda: FloatParameter("p", "0.1", 1.0),  # no string-to-number parsing
        lambda: FloatParameter("p", 0.1, 1.0, log="yes"),
        lambda: FloatParameter(None, 0.1, 1.0),
    ],
    ids=[
        "int-given-float",
        "int-given-bool",
        "float-given-str",
        "log-given-str",
        "name-none",
    ],
)
def test_wrong_types_are_rejected(build: Callable[[], object]) -> None:
    # Strict pydantic validation; a plain dataclass stored every one of these.
    with pytest.raises(ValueError, match="validation error"):
        build()


def test_float_parameter_accepts_int_bounds() -> None:
    # The one conversion strict mode allows, as Optuna does.
    parameter = FloatParameter("p", 0, 1)
    assert type(parameter.low) is float and type(parameter.high) is float


def test_categorical_choices_reach_optuna_in_order() -> None:
    # Every type Optuna can store, mixed in one parameter.
    choices = ("gbdt", None, True, 2, 0.5)
    parameter = CategoricalParameter("p", choices)
    trial = _trial()

    value = parameter.suggest(trial)

    assert trial.distributions["p"] == CategoricalDistribution(choices)
    assert value in choices


@pytest.mark.parametrize(
    "choices",
    [("gbdt", "dart", "gbdt"), (True, 1)],
    ids=["repeated", "true-equals-1"],
)
def test_duplicate_choices_are_rejected(choices: tuple[object, ...]) -> None:
    # Optuna maps 1 to True's slot, so (True, 1) is a duplicate there too.
    with pytest.raises(ValueError, match=r"parameter 'p': choices must be unique"):
        CategoricalParameter("p", choices)  # type: ignore[arg-type]


def test_non_scalar_choice_is_rejected() -> None:
    # Optuna would only warn; pydantic's type check rejects it first.
    with pytest.raises(ValueError, match="validation errors? for CategoricalParameter"):
        CategoricalParameter("p", ([1], "b"))  # type: ignore[arg-type]


def test_list_of_choices_is_stored_as_a_tuple() -> None:
    # A list is the natural way to write choices, but it could be changed
    # after creation; the stored tuple cannot.
    parameter = CategoricalParameter("p", ["gbdt", "dart"])  # type: ignore[arg-type]

    assert parameter.choices == ("gbdt", "dart")
    with pytest.raises(FrozenInstanceError):
        parameter.choices = ("goss",)  # type: ignore[misc]


@pytest.mark.parametrize(
    "choice", [np.int64(7), np.True_], ids=["numpy-int", "numpy-bool"]
)
def test_numpy_integer_choices_are_rejected(choice: object) -> None:
    # Strict mode alone would store these as 7.0 and 1.0.
    with pytest.raises(ValueError, match="choices must be None, bool, int"):
        CategoricalParameter("p", (choice, "a"))  # type: ignore[arg-type]


def test_numpy_float_choice_becomes_a_plain_float() -> None:
    # np.float64 is a float subclass, so its value survives unchanged.
    parameter = CategoricalParameter("p", (np.float64(0.5), "a"))
    assert parameter.choices == (0.5, "a")
    assert type(parameter.choices[0]) is float


def test_empty_choices_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"parameter 'p': .*one or more"):
        CategoricalParameter("p", ())


def test_lightgbm_parameters_run_in_a_study() -> None:
    # The DirectCohortModel bounds ([direct_cohort_search_space]) as a plain list.
    parameters = [
        IntParameter("model__max_depth", 3, 8),
        IntParameter("model__num_leaves", 7, 63),
        IntParameter("model__min_child_samples", 5, 40),
        FloatParameter("model__learning_rate", 0.01, 0.2, log=True),
        IntParameter("model__n_estimators", 50, 400),
        FloatParameter("model__reg_alpha", 1e-8, 10.0, log=True),
        FloatParameter("model__reg_lambda", 1e-8, 10.0, log=True),
        FloatParameter("model__min_split_gain", 0.0, 1.0),
        FloatParameter("model__subsample", 0.7, 1.0),
        FloatParameter("model__colsample_bytree", 0.7, 1.0),
    ]

    def objective(trial: optuna.Trial) -> float:
        return float(len({p.name: p.suggest(trial) for p in parameters}))

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    assert [trial.state.name for trial in study.trials] == ["COMPLETE"] * 5
    assert all(len(trial.params) == 10 for trial in study.trials)


def test_package_exports_the_public_names() -> None:
    assert set(tuning.__all__) == {
        "CategoricalParameter",
        "FloatParameter",
        "IntParameter",
        "ParamValue",
        "Parameter",
    }
