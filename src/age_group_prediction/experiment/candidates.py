"""Ordered enumeration of the predeclared candidate feature specifications.

Section 5.3 compares a small predeclared sequence rather than an unrestricted
feature search, and requires that candidate factories "enumerate an ordered set
of specs; they do not fit models or select a winner". Building the list by hand
at each call site gives no guarantee that the declared families are the ones
actually compared, nor that they are compared in a stable order.

Nothing here fits, scores, or ranks anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..modeling_config import (
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    FeatureInteraction,
    FeatureSpec,
    SesForm,
)

__all__ = [
    "FeatureSpecCandidate",
    "enumerate_feature_specs",
]

# Section 5.3, in the order it declares them. The base form comes first so a
# comparison always contains the parsimonious specification it is measured
# against.
_SES_FORMS: tuple[SesForm, ...] = ("linear", "quadratic", "spline")

_COMPONENT_BASES = {
    "tree": DEFAULT_TREE_FEATURE_SPEC,
    "total_count": DEFAULT_TOTAL_FEATURE_SPEC,
    "composition": DEFAULT_PROBABILITY_FEATURE_SPEC,
}

# Room-composition interactions are component-specific by construction: the
# household-size term is declared for total counts and the median-age term for
# age-group probabilities. `FeatureSpec` rejects the wrong pairing, so the
# enumeration must not offer it.
_INTERACTIONS_BY_COMPONENT: dict[str, tuple[FeatureInteraction, ...]] = {
    "tree": (),
    "total_count": (
        "ses_x_household_size",
        "daycare_x_median_age",
        "room_share_x_household_size",
    ),
    "composition": (
        "ses_x_household_size",
        "daycare_x_median_age",
        "room_share_x_median_age",
    ),
}


@dataclass(frozen=True)
class FeatureSpecCandidate:
    """One predeclared specification and the stable name identifying it."""

    name: str
    spec: FeatureSpec

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature spec candidates require a name")


def enumerate_feature_specs(
    component: str,
    *,
    include_daycare_saturation: bool = False,
    interactions: Sequence[FeatureInteraction] | None = None,
) -> tuple[FeatureSpecCandidate, ...]:
    """Enumerate section 5.3's predeclared specs for one component, in order.

    Tree candidates take no interaction or transform variants: the gradient
    boosting model represents them itself, so enumerating them would compare
    the same hypothesis repeatedly.

    ``include_daycare_saturation`` adds section 5.3's sixth family, the
    low-degree daycare saturation curve, which that section admits only when
    held-out residuals justify it. It is therefore opt-in rather than default.
    """
    if component not in _COMPONENT_BASES:
        raise ValueError(
            f"Unknown feature component: {component!r}; "
            f"expected one of {sorted(_COMPONENT_BASES)}"
        )
    base = _COMPONENT_BASES[component]

    available = _INTERACTIONS_BY_COMPONENT[component]
    if interactions is None:
        selected_interactions = available
    else:
        unknown = set(interactions).difference(available)
        if unknown:
            raise ValueError(
                f"Interactions {sorted(unknown)} are not declared for component "
                f"{component!r}; available: {sorted(available)}"
            )
        selected_interactions = tuple(
            interaction for interaction in available if interaction in set(interactions)
        )

    candidates: list[FeatureSpecCandidate] = []

    # Family 1: the SES forms, compared against each other.
    for ses_form in _SES_FORMS:
        candidates.append(
            FeatureSpecCandidate(
                name=f"{component}__ses_{ses_form}",
                spec=replace(base, ses_form=ses_form),
            )
        )

    # Families 2 to 5: one interaction at a time on the linear SES base, so
    # each term is attributable. Every interaction carries its main effects,
    # which `FeatureSpec` enforces.
    for interaction in selected_interactions:
        candidates.append(
            FeatureSpecCandidate(
                name=f"{component}__{interaction}",
                spec=replace(base, interactions=(interaction,)),
            )
        )

    # Family 6: the daycare saturation curve, opt-in.
    if include_daycare_saturation:
        candidates.append(
            FeatureSpecCandidate(
                name=f"{component}__daycare_log1p",
                spec=replace(base, daycare_form="log1p"),
            )
        )

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Enumerated feature spec names must be unique")
    return tuple(candidates)
