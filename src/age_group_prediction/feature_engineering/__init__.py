"""Typed feature transformations and the design matrices built from them.

Each transformation is its own model carrying exactly the parameters it needs,
so a missing or nonsensical parameter is a construction-time error rather than
a failure deep inside a fold. :mod:`transforms` holds those transformations;
:mod:`transformer` groups them into named plans, applies each to its
columns and fits them.
"""

from .transformer import FeatureTransformer, Interaction, Plan
from .transforms import (
    Center,
    CenterByReferencePoint,
    DomainMinMax,
    DomainScale,
    Log,
    Log1p,
    Log1pRatioScaler,
    OneHot,
    Quadratic,
    RelativeSaturation,
    Standardize,
    Transform,
)

__all__ = [
    "Center",
    "CenterByReferencePoint",
    "DomainMinMax",
    "DomainScale",
    "FeatureTransformer",
    "Interaction",
    "Log",
    "Log1p",
    "Log1pRatioScaler",
    "OneHot",
    "Plan",
    "Quadratic",
    "RelativeSaturation",
    "Standardize",
    "Transform",
]
