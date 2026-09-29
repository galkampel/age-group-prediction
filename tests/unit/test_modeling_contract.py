"""Every shipped model keeps scikit-learn's parameter rules, so a tuner can clone it.

A model that converts, validates or renames a constructor argument, sets
state in ``__init__``, or overrides ``set_params`` wrongly breaks the tuner's
``clone(model).set_params(**params)`` silently. Of sklearn's data-free
checks, these four are the smallest set that caught each such mistake when
tried on deliberately broken models. An ``__init__`` that copies a mutable
argument passes them all when built with defaults, so each model is checked on
an example instance instead, which also lets a model have required arguments.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import (
    check_do_not_raise_errors_in_init_or_set_params,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from age_group_prediction.modeling import BaseAgeGroupModel, DirectCohortModel


def _models(base: type) -> list[type[BaseAgeGroupModel]]:
    """Every concrete subclass shipped in ``modeling``, however deeply derived.

    Filtered by package: ``__subclasses__`` also returns test stand-ins. A
    class reached through two parents is listed once.
    """
    found: dict[type[BaseAgeGroupModel], None] = {}
    for subclass in base.__subclasses__():
        package = subclass.__module__.split(".")[:2]
        if package == ["age_group_prediction", "modeling"] and not (
            inspect.isabstract(subclass)
        ):
            found[subclass] = None
        found.update(dict.fromkeys(_models(subclass)))
    return list(found)


MODELS = _models(BaseAgeGroupModel)

# An unfitted model and data it can be fitted on, in the shapes it takes.
type Example = tuple[BaseAgeGroupModel, pd.DataFrame, pd.Series | pd.DataFrame]


def _direct_cohort_example() -> Example:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.Series(rng.poisson(np.exp(1 + 0.5 * X["x"])))
    return DirectCohortModel(), X, y


# Factories, so every test gets its own model and data and none is built at import.
EXAMPLES: dict[type[BaseAgeGroupModel], Callable[[], Example]] = {
    DirectCohortModel: _direct_cohort_example,
}

CHECKS: list[Callable[[str, BaseAgeGroupModel], None]] = [
    check_do_not_raise_errors_in_init_or_set_params,  # converts or validates in __init__
    check_parameters_default_constructible,  # replaces a default
    check_no_attributes_set_in_init,  # sets state in __init__
    check_set_params,  # a set_params that drops or converts values
]


def test_every_shipped_model_has_an_example() -> None:
    # A model without an example would go unchecked, and so would one whose
    # example builds another class (a copied factory). Equality with a non-empty
    # EXAMPLES also fails an empty discovery, which pytest would only skip.
    assert set(MODELS) == set(EXAMPLES), (
        f"discovered {sorted(c.__name__ for c in MODELS)}, "
        f"but examples exist for {sorted(c.__name__ for c in EXAMPLES)}"
    )
    for model_class, example in EXAMPLES.items():
        model, _, _ = example()
        assert type(model) is model_class, f"{model_class.__name__}'s example"


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("check", CHECKS, ids=lambda f: f.__name__)
def test_model_keeps_the_parameter_contract(
    model_class: type[BaseAgeGroupModel],
    check: Callable[[str, BaseAgeGroupModel], None],
) -> None:
    model, _, _ = EXAMPLES[model_class]()
    check(model_class.__name__, model)


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
def test_a_refit_equals_a_fresh_fit(model_class: type[BaseAgeGroupModel]) -> None:
    # The tuner reuses one copy across a trial's folds, so fit must replace
    # all fitted state rather than build on the previous fold's.
    template, X, y = EXAMPLES[model_class]()
    half = len(X) // 2
    # Positional, for a Series or a DataFrame y alike.
    first, second = slice(0, half), slice(half, None)

    refitted = (
        clone(template)
        .fit(X.iloc[first], y.iloc[first])
        .fit(X.iloc[second], y.iloc[second])
    )
    fresh = clone(template).fit(X.iloc[second], y.iloc[second])

    np.testing.assert_array_equal(refitted.predict(X), fresh.predict(X))
