"""The values each hyperparameter may take: one class per Optuna ``suggest_*`` call.

``name`` is the ``estimator.set_params`` key, so a ``Pipeline`` step's
parameter is ``"<step>__<parameter>"``.

A parameter is validated when created, not at the first trial: pydantic checks
the types strictly (``IntParameter("p", 1.5, 9)`` fails), then Optuna's own
distribution class checks the values. Both raise ``ValueError``. Pass Python
numbers: numpy integers are rejected (use ``int(x)``), not silently converted.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Annotated

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import BaseTrial
from pydantic import BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

__all__ = [
    "CategoricalParameter",
    "FloatParameter",
    "IntParameter",
    "ParamValue",
    "Parameter",
]

type ParamValue = None | bool | int | float | str  # the types Optuna can store

# Strict: no silent conversion, so "0.1" is not read as 0.1, nor 1 as True.
# NaN and inf bounds would otherwise pass here and fail only at the first trial.
_STRICT = ConfigDict(strict=True, allow_inf_nan=False)

_SCALARS = (type(None), bool, int, float, str)


def _as_choice_tuple(value: object) -> object:
    """Accept a list of choices as a tuple; reject numpy integers and bools.

    Strict mode alone turns ``np.int64(7)`` into ``7.0`` and ``np.True_`` into
    ``1.0``, which would hand LightGBM ``num_leaves=7.0``.
    """
    if isinstance(value, list):
        value = tuple(value)
    # ValueError, not TypeError: pydantic reports only the former as a
    # validation error.
    rejected = (
        [c for c in value if not isinstance(c, _SCALARS)]
        if isinstance(value, tuple)
        else []
    )
    if rejected:
        raise ValueError(
            f"choices must be None, bool, int, float or str, got {rejected!r}"
        )
    return value


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
    def suggest(self, trial: BaseTrial) -> ParamValue:
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


@pydantic_dataclass(frozen=True, config=_STRICT)
class CategoricalParameter(Parameter):
    """One of ``choices``, drawn by ``trial.suggest_categorical``."""

    # A list is accepted and stored as an immutable tuple.
    choices: Annotated[tuple[ParamValue, ...], BeforeValidator(_as_choice_tuple)]

    def __post_init__(self) -> None:
        super().__post_init__()
        # Optuna accepts duplicates silently, giving that choice double weight.
        # A set also merges True, 1 and 1.0, which Optuna cannot tell apart.
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(
                f"parameter {self.name!r}: choices must be unique, got {self.choices}"
            )

    def suggest(self, trial: BaseTrial) -> ParamValue:
        """Draw a value for ``trial``."""
        return trial.suggest_categorical(self.name, self.choices)

    def _distribution(self) -> CategoricalDistribution:
        return CategoricalDistribution(self.choices)
