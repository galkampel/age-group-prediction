"""Every shipped model keeps scikit-learn's parameter rules, so a tuner can clone it.

A model that converts, validates or renames a constructor argument, sets
state in ``__init__``, or overrides ``set_params`` wrongly breaks the tuner's
``clone(model).set_params(**params)`` silently. Of sklearn's data-free
checks, these four are the smallest set that caught each such mistake when
tried on deliberately broken models. An ``__init__`` that copies a mutable
argument passes them all when built with defaults, so each model is checked on
an example instance instead, which also lets a model have required arguments.

The package must also stay independent of the old stack it replaces.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from importlib.util import resolve_name
from pathlib import Path

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

import age_group_prediction.modeling as modeling_package
from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    DirectCohortModel,
    IndependentCohortModels,
    ModelPipeline,
    TotalChildrenModel,
)


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


def _model_pipeline_example() -> Example:
    # No exposure: the refit test calls fit(X, y).
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.Series(rng.poisson(np.exp(1 + 0.5 * X["x"])))
    features = FeatureTransformer(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    return ModelPipeline(features, DirectCohortModel()), X, y


def _independent_cohorts_example() -> Example:
    # No exposure: the refit test calls fit(X, y).
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.DataFrame(
        {
            "a": rng.poisson(np.exp(1 + 0.5 * X["x"])),
            "b": rng.poisson(np.exp(0.5 - 0.3 * X["x"])),
        }
    )
    features = FeatureTransformer(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    model = IndependentCohortModels(
        {cohort: ModelPipeline(features, DirectCohortModel()) for cohort in y}
    )
    return model, X, y


def _total_children_example() -> Example:
    # No exposure: the refit test calls fit(X, y), so the default offset would raise.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.Series(rng.poisson(np.exp(1 + 0.5 * X["x"])))
    return TotalChildrenModel(use_exposure=False), X, y


def _cohort_probability_example() -> Example:
    # Every building needs a child: the shares are the observation.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.DataFrame(
        {
            "a": 1 + rng.poisson(np.exp(1 + 0.5 * X["x"])),
            "b": rng.poisson(np.exp(0.5 - 0.3 * X["x"])),
        }
    )
    return CohortProbabilityModel(), X, y


# Factories, so every test gets its own model and data and none is built at import.
EXAMPLES: dict[type[BaseAgeGroupModel], Callable[[], Example]] = {
    DirectCohortModel: _direct_cohort_example,
    ModelPipeline: _model_pipeline_example,
    IndependentCohortModels: _independent_cohorts_example,
    TotalChildrenModel: _total_children_example,
    CohortProbabilityModel: _cohort_probability_example,
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


# The old stack (MULTI_COHORT_MODELS_PLAN.md §4), deleted once every model is rebuilt.
OLD_STACK = frozenset(
    {
        "data_splitting",
        "distributions",
        "evaluation",
        "fitted_features",
        "metrics",
        "modeling_config",
        "models",
        "predictive",
        "resampling",
        "results",
        "state_bundle",
        "tuning",
    }
)


def _imported_modules(path: Path, package: str) -> list[str]:
    """Every absolute module name an import in ``path`` may reach."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            module = resolve_name(relative, package) if node.level else relative
            # `from .. import models` names the module among the imported names.
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_modeling_never_imports_the_old_stack() -> None:
    # Read from the source, not sys.modules: the package root imports the old
    # stack, so at runtime every module would look dependent on it. Static
    # imports only; an importlib call by name would pass unseen.
    modeling_dir = Path(modeling_package.__file__).parent
    offenders: list[str] = []
    for path in sorted(modeling_dir.rglob("*.py")):
        subpackages = path.parent.relative_to(modeling_dir.parent).parts
        package = ".".join(("age_group_prediction", *subpackages))
        for name in _imported_modules(path, package):
            if name.startswith("age_group_prediction.") and (
                name.split(".")[1] in OLD_STACK
            ):
                offenders.append(f"{path.name}: {name}")
    assert offenders == []
