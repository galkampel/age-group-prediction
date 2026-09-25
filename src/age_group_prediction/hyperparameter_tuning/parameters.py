"""The values each hyperparameter may take: one class per Optuna ``suggest_*`` call.

``name`` is the model's ``set_params`` key, e.g. ``"learning_rate"``.

A parameter is validated when created, not at the first trial: pydantic checks
the types strictly (``IntParameter("p", 1.5, 9)`` fails), then Optuna's own
distribution class checks the values. Both raise ``ValueError``. Pass Python
numbers as bounds: numpy integers are rejected (use ``int(x)``), not silently
converted.

Categorical choices may be any objects, e.g. layer sizes ``[[32], [64, 32]]``,
and are passed to Optuna unchanged. Optuna warns at each ``suggest`` that
non-scalar choices suit only in-memory storage: a persistent storage keeps
them as JSON, so a tuple comes back as a list.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Annotated, Any

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import BaseTrial
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._config import _STRICT

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


@pydantic_dataclass(frozen=True, config=_STRICT)
class CategoricalParameter(Parameter):
    """One of ``choices``, drawn by ``trial.suggest_categorical``."""

    # Any objects, left as given. A list is stored as an immutable tuple; a
    # set is rejected, since its order, and so a seeded study, varies by run.
    choices: Annotated[
        tuple[Any, ...],
        BeforeValidator(lambda v: tuple(v) if isinstance(v, list) else v),
    ]

    def __post_init__(self) -> None:
        super().__post_init__()
        # Optuna accepts duplicates silently, giving that choice double weight.
        # Compared with ==, which handles unhashable choices such as lists and,
        # like Optuna, treats True, 1 and 1.0 as one choice.
        c = self.choices
        if any(a == b for i, a in enumerate(c) for b in c[i + 1 :]):
            raise ValueError(
                f"parameter {self.name!r}: choices must be unique, got {self.choices}"
            )

    def suggest(self, trial: BaseTrial) -> Any:
        """Draw a value for ``trial``."""
        return trial.suggest_categorical(self.name, self.choices)

    def _distribution(self) -> CategoricalDistribution:
        with warnings.catch_warnings():
            # Non-scalar choices are allowed. Optuna's note that they suit only
            # in-memory storage is left to each suggest, not made an error here.
            warnings.filterwarnings(
                "ignore", message="Choices for a categorical distribution"
            )
            return CategoricalDistribution(self.choices)
