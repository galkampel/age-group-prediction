"""Typed feature transformations and the design matrices built from them.

Each transformation is its own model carrying exactly the parameters it needs,
so a missing or nonsensical parameter is a construction-time error rather than
a failure deep inside a fold. :mod:`transforms` holds those transformations;
:mod:`transformer` groups them into named column plans and fits them.
"""

from .transformer import ColumnPlan, FeatureTransformer, Interaction
from .transforms import (
    Center,
    DomainMinMax,
    DomainScale,
    Log,
    Log1p,
    OneHot,
    Quadratic,
    RelativeSaturation,
    Standardize,
    Transform,
)

__all__ = [
    "Center",
    "ColumnPlan",
    "DomainMinMax",
    "DomainScale",
    "FeatureTransformer",
    "Interaction",
    "Log",
    "Log1p",
    "OneHot",
    "Quadratic",
    "RelativeSaturation",
    "Standardize",
    "Transform",
]
