"""The values each hyperparameter may take: one class per Optuna ``suggest_*`` call.

``name`` is the model's ``set_params`` key, e.g. ``"estimator__learning_rate"``.

A parameter is validated when created, not at the first trial: pydantic checks
the types strictly (``IntParameter("p", 1.5, 9)`` fails), then Optuna's own
distribution class checks the values. Both raise ``ValueError``. Pass Python
numbers as bounds: numpy integers are rejected (use ``int(x)``), not silently
converted.

Categorical choices are Optuna's scalar choice types, ``None``, ``bool``,
``int``, ``float`` and ``str``, so a trial's parameters are immutable, plain
Python values; a numpy integer is rejected, like a numpy bound.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from math import isfinite
from typing import Annotated, Any, get_args

from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import BaseTrial
from pydantic import AfterValidator, BeforeValidator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._config import _STRICT

# (NoneType, bool, int, float, str), from Optuna's own alias.
_CHOICE_TYPES = get_args(CategoricalChoiceType)

__all__ = [
    "CategoricalParameter",
    "FloatParameter",
    "IntParameter",
    "Parameter",
]


@pydantic_dataclass(frozen=True, config=_STRICT)
class Parameter(ABC):
    """One hyperparameter: its ``set_params`` name and the values it may take."""

    name: str

    def __post_init__(self) -> None:
        # Optuna only warns about some bad settings: a step that does not divide
        # the range makes it quietly lower `high`. Its UserWarnings are raised
        # as errors; other warnings pass through untouched. The filter change is
        # process-wide while it lasts, so building parameters is not thread-safe.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                self._distribution()
            except (ValueError, UserWarning) as error:
                raise ValueError(f"parameter {self.name!r}: {error}") from error

    @abstractmethod
    def suggest(self, trial: BaseTrial) -> Any:
        """Draw this parameter's value for ``trial``."""

    @abstractmethod
    def _distribution(self) -> BaseDistribution:
        """The Optuna distribution :meth:`suggest` samples from."""


@pydantic_dataclass(frozen=True, config=_STRICT)
class FloatParameter(Parameter):
    """A float in ``[low, high]``, drawn by ``trial.suggest_float``.

    ``log=True`` samples evenly in log space and needs ``low > 0``. ``step``
    must divide ``high - low``. The two cannot be combined.
    """

    low: float
    high: float
    _: KW_ONLY  # log and step are keyword-only, as in suggest_float
    log: bool = False
    step: float | None = None

    def suggest(self, trial: BaseTrial) -> float:
        """Draw a value for ``trial``."""
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def _distribution(self) -> FloatDistribution:
        return FloatDistribution(self.low, self.high, log=self.log, step=self.step)


@pydantic_dataclass(frozen=True, config=_STRICT)
class IntParameter(Parameter):
    """An integer in ``[low, high]``, drawn by ``trial.suggest_int``.

    ``log=True`` samples evenly in log space and needs ``low >= 1`` and
    ``step == 1``. ``step`` must divide ``high - low``.
    """

    low: int
    high: int
    _: KW_ONLY  # log and step are keyword-only, as in suggest_int
    log: bool = False
    step: int = 1

    def suggest(self, trial: BaseTrial) -> int:
        """Draw a value for ``trial``."""
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def _distribution(self) -> IntDistribution:
        return IntDistribution(self.low, self.high, log=self.log, step=self.step)


def _scalar_choices(choices: tuple[Any, ...]) -> tuple[CategoricalChoiceType, ...]:
    # Optuna's declared choice types; it only warns about others, which it
    # would hand back by reference (a list choice edited through a result
    # edits the study) and which json cannot write. isinstance, not a strict
    # pydantic union: that would turn np.int64(7) into 7.0 silently.
    unsupported = [c for c in choices if not isinstance(c, _CHOICE_TYPES)]
    if unsupported:
        raise ValueError(
            f"choices must be None, bool, int, float or str, got {unsupported}"
        )
    # NaN or inf would reach best_params, which strict json cannot write.
    if any(isinstance(c, float) and not isfinite(c) for c in choices):
        raise ValueError(f"float choices must be finite, got {choices}")
    return choices


@pydantic_dataclass(frozen=True, config=_STRICT)
class CategoricalParameter(Parameter):
    """One of ``choices``, drawn by ``trial.suggest_categorical``.

    Choices are ``None``, ``bool``, ``int``, ``float`` or ``str`` (Optuna's
    own types), left as given.
    """

    # A list is stored as an immutable tuple; a set is rejected, since its
    # order, and so a seeded study, varies by run.
    choices: Annotated[
        tuple[Any, ...],
        BeforeValidator(lambda v: tuple(v) if isinstance(v, list) else v),
        AfterValidator(_scalar_choices),
    ]

    def __post_init__(self) -> None:
        super().__post_init__()
        # Optuna accepts duplicates silently, giving that choice double weight.
        # A set compares by ==, which, like Optuna, treats True, 1 and 1.0 as
        # one choice.
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(
                f"parameter {self.name!r}: choices must be unique, got {self.choices}"
            )

    def suggest(self, trial: BaseTrial) -> Any:
        """Draw a value for ``trial``."""
        return trial.suggest_categorical(self.name, self.choices)

    def _distribution(self) -> CategoricalDistribution:
        return CategoricalDistribution(self.choices)
