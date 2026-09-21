"""Tests for declaring a design matrix: column plans and their validation.

A malformed declaration should be refused before any data is involved, so most
of these assert that something raises. Per-plan rules fail when the plan is
built; rules that span plans fail from ``validate()``, which ``fit()`` calls.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    DomainMinMax,
    DomainScale,
    FeatureTransformer,
    Interaction,
    Log,
    Log1p,
    OneHot,
    Quadratic,
    RelativeSaturation,
    Standardize,
)

# One instance per kind, to check a plan accepts the whole vocabulary. Kept
# here rather than shared with test_transforms.py: that module's list exists to
# cover serialization round-trips, and the two would drift apart the moment
# either purpose changed.
_ONE_OF_EACH_TRANSFORM = (
    Standardize(),
    Center(),
    Quadratic(),
    Log(),
    Log1p(),
    DomainScale(scale=0.1),
    DomainMinMax(minimum=0.0, maximum=8.0),
    RelativeSaturation(),
    OneHot(categories=("none", "existing", "planned"), reference_category="none"),
)


def _plan(name: str, *columns: str, transforms=()) -> ColumnPlan:
    return ColumnPlan(name=name, columns=columns, transforms=transforms)


# --- ColumnPlan: building one group's chain ----------------------------------


def test_a_plan_without_transforms_is_a_passthrough() -> None:
    plan = ColumnPlan(name="ses_raw", columns=("ses",))
    assert plan.build() == "passthrough"


def test_a_single_transform_needs_no_pipeline_wrapper() -> None:
    plan = ColumnPlan(name="ses_z", columns=("ses",), transforms=(Standardize(),))
    assert isinstance(plan.build(), StandardScaler)


def test_a_chain_builds_a_pipeline_in_declaration_order() -> None:
    plan = ColumnPlan(
        name="room_share",
        columns=("3_rooms_share",),
        transforms=(Center(), DomainScale(scale=0.1)),
    )
    built = plan.build()
    assert isinstance(built, Pipeline)
    assert [name for name, _ in built.steps] == ["0_center", "1_domain_scale"]


def test_a_chain_may_repeat_a_kind() -> None:
    # Index-prefixed step names; a bare kind would collide and sklearn would
    # reject the Pipeline.
    plan = ColumnPlan(name="twice", columns=("ses",), transforms=(Center(), Center()))
    assert [name for name, _ in plan.build().steps] == ["0_center", "1_center"]


def test_a_room_share_chain_centers_then_rescales() -> None:
    df = pd.DataFrame({"3_rooms_share": [0.1, 0.2, 0.3]})
    plan = ColumnPlan(
        name="room_share",
        columns=("3_rooms_share",),
        transforms=(Center(), DomainScale(scale=0.1)),
    )
    transformed = plan.build().fit_transform(df)
    # (x - 0.2) / 0.1, so the coefficient reads per 10 percentage points.
    np.testing.assert_allclose(transformed.ravel(), [-1.0, 0.0, 1.0], atol=1e-12)


def test_a_chain_names_its_output_for_the_last_renaming_step() -> None:
    plan = ColumnPlan(
        name="ses_z_sq", columns=("ses",), transforms=(Standardize(), Quadratic())
    )
    built = plan.build().fit(pd.DataFrame({"ses": [1.0, 2.0, 3.0]}))
    assert list(built.get_feature_names_out()) == ["ses_squared"]


def test_a_plan_accepts_every_transform_in_the_vocabulary() -> None:
    # Each in its own plan, since a one-hot may not share one.
    for index, transform in enumerate(_ONE_OF_EACH_TRANSFORM):
        plan = _plan(f"p{index}", f"c{index}", transforms=(transform,))
        assert plan.transforms == (transform,)


# --- ColumnPlan: what cannot be declared -------------------------------------


def test_a_plan_rejects_an_empty_name_or_no_columns() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        ColumnPlan(name="", columns=("ses",))
    with pytest.raises(ValidationError, match="at least 1 item"):
        ColumnPlan(name="empty", columns=())


def test_a_plan_named_remainder_is_rejected_when_fitted() -> None:
    # Not checked here: "remainder" is ColumnTransformer's own entry name and
    # it refuses the clash itself. Pinned because the guarantee is now its.
    transformer = FeatureTransformer(
        plans=(_plan("remainder", "ses", transforms=(Standardize(),)),)
    )
    with pytest.raises(ValueError, match="remainder"):
        transformer.fit(pd.DataFrame({"ses": [1.0, 2.0, 3.0]}))


def test_a_single_column_may_be_given_as_a_bare_string() -> None:
    # The common case, so it should not require a one-element sequence.
    assert ColumnPlan(name="ses_z", columns="ses").columns == ("ses",)


# --- Delegated: mistakes nothing here checks, because something else does ----
#
# Each was once refused when the plan or transformer was built. The check went
# because the failure is loud either way; what is pinned now is that the error
# still arrives and still names the culprit.


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ses": [1.0, 2.0, 3.0],
            "school_status": ["none", "existing", "planned"],
            "zone": ["a", "b", "a"],
        }
    )


def test_repeated_columns_in_one_plan_are_rejected_when_fitted() -> None:
    with pytest.raises(ValueError, match="unique column names"):
        FeatureTransformer(plans=(_plan("dup", "ses", "ses"),)).fit(_frame())


def test_a_one_hot_over_two_columns_is_rejected_when_fitted() -> None:
    # The spec carries one categories list and one reference, so it describes
    # a single column; two categoricals belong in two plans.
    encoder = OneHot(categories=("none", "existing"), reference_category="none")
    with pytest.raises(ValueError, match="Shape mismatch"):
        FeatureTransformer(
            plans=(_plan("cats", "school_status", "zone", transforms=(encoder,)),)
        ).fit(_frame())


def test_a_one_hot_chained_with_another_transform_is_rejected_when_fitted() -> None:
    encoder = OneHot(categories=("none", "existing"), reference_category="none")
    with pytest.raises(ValueError, match="could not convert string to float"):
        FeatureTransformer(
            plans=(_plan("odd", "school_status", transforms=(Standardize(), encoder)),)
        ).fit(_frame())


def test_a_missing_planned_column_is_rejected_when_fitted() -> None:
    with pytest.raises(ValueError, match="absent"):
        FeatureTransformer(
            plans=(_plan("nope", "absent", transforms=(Standardize(),)),)
        ).fit(_frame())


def test_colliding_output_names_are_rejected_when_fitted() -> None:
    with pytest.raises(ValueError, match="are not unique"):
        FeatureTransformer(
            plans=(
                _plan("ses_z", "ses", transforms=(Standardize(),)),
                _plan("ses_centered", "ses", transforms=(Center(),)),
            )
        ).fit(_frame())


@pytest.mark.filterwarnings("ignore:.*encountered in log:RuntimeWarning")
def test_a_log_after_another_transform_is_caught_by_the_output_check() -> None:
    # Centering first makes values non-positive, so the log yields -inf. The
    # chain is no longer forbidden outright; the output check catches it.
    with pytest.raises(ValueError, match="non-finite"):
        FeatureTransformer(
            plans=(_plan("c", "ses", transforms=(Center(), Log())),)
        ).fit_transform(_frame())


def test_a_plan_round_trips_with_its_transform_subclasses_intact() -> None:
    plan = ColumnPlan(
        name="daycare",
        columns=("n_daycares_500m",),
        transforms=(DomainMinMax(minimum=0.0, maximum=8.0),),
    )
    restored = ColumnPlan.model_validate(plan.model_dump())
    assert restored == plan
    assert isinstance(restored.transforms[0], DomainMinMax)


# --- FeatureTransformer: rules that span plans -------------------------------


def _transformer(**overrides) -> FeatureTransformer:
    defaults = {"plans": (_plan("ses_z", "ses", transforms=(Standardize(),)),)}
    return FeatureTransformer(**{**defaults, **overrides})


def test_a_declaration_error_is_raised_by_validate_not_by_construction() -> None:
    # set_params assigns attributes without re-entering __init__, so a check
    # placed there would be skipped by GridSearchCV. validate() is the
    # chokepoint, and fit() calls it -- so constructing an invalid declaration
    # must succeed and only validate() may refuse it.
    transformer = FeatureTransformer(plans=(), remainder="drop")
    with pytest.raises(ValueError, match="no columns"):
        transformer.validate()


def test_duplicate_plan_names_are_rejected_when_fitted() -> None:
    # Not checked in validate(): ColumnTransformer refuses duplicate entry
    # names itself. Pinned because the guarantee is now its.
    transformer = FeatureTransformer(
        plans=(
            _plan("ses_z", "ses", transforms=(Standardize(),)),
            _plan("ses_z", "median_age", transforms=(Standardize(),)),
        )
    )
    with pytest.raises(ValueError, match="not unique"):
        transformer.fit(pd.DataFrame({"ses": [1.0, 2.0], "median_age": [3.0, 4.0]}))


def test_one_column_may_feed_several_plans() -> None:
    # This is how the SES-quadratic variant adds ses_squared beside ses; it must
    # stay legal, and it is legal precisely because the two emit different
    # columns.
    transformer = _transformer(
        plans=(
            _plan("ses_z", "ses", transforms=(Standardize(),)),
            _plan("ses_z_sq", "ses", transforms=(Standardize(), Quadratic())),
        )
    )
    names = transformer.fit(pd.DataFrame({"ses": [1.0, 2.0, 3.0]}))
    assert list(names.get_feature_names_out()) == ["ses", "ses_squared"]


def test_dropping_everything_needs_at_least_one_plan() -> None:
    # An empty ColumnTransformer with remainder="drop" is legal for sklearn and
    # yields a design matrix with zero columns, which is never intended.
    with pytest.raises(ValueError, match="no columns"):
        FeatureTransformer(plans=(), remainder="drop").validate()


def test_an_empty_passthrough_transformer_is_meaningful() -> None:
    # Every column, untouched, is a real configuration.
    FeatureTransformer(plans=(), remainder="passthrough").validate()


def test_an_exposure_may_also_be_a_predictor() -> None:
    # Model A keeps n_apartments as a feature and passes log n as LightGBM's
    # init_score (FEATURE_TRANSFORMATIONS.md 3.3): the offset asserts exact
    # proportionality and the feature lets the trees learn departures from it.
    # Whether that is wanted is the caller's modeling choice, not this class's.
    transformer = FeatureTransformer(
        plans=(_plan("apartments", "n_apartments"),),
        exposure_column="n_apartments",
    )
    df = pd.DataFrame({"n_apartments": [12.0, 40.0, 80.0]})
    assert list(transformer.fit_transform(df).columns) == ["n_apartments"]
    offset = transformer.log_exposure(df)
    assert offset.name == "log_n_apartments"
    np.testing.assert_allclose(offset.to_numpy(), np.log([12.0, 40.0, 80.0]))


# --- Interactions: what can be declared --------------------------------------
#
# An Interaction is to the interaction step what a ColumnPlan is to the base
# matrix: it names the columns, and a transform does the arithmetic. Its sides
# name design-matrix columns -- what a plan emits, not what it consumes --
# because the step runs on the plans' output.


def _interacting_transformer(
    interactions: Sequence[Interaction] = (),
) -> FeatureTransformer:
    """A transformer whose base matrix is ses, ses_squared, 3_rooms_share."""
    return FeatureTransformer(
        plans=(
            _plan("ses_z", "ses", transforms=(Standardize(),)),
            _plan("ses_sq", "ses", transforms=(Standardize(), Quadratic())),
            _plan("room_share", "3_rooms_share", transforms=(Center(),)),
        ),
        interactions=interactions,
    )


def _interaction_frame() -> pd.DataFrame:
    return pd.DataFrame({"ses": [1.0, 2.0, 3.0], "3_rooms_share": [0.1, 0.2, 0.3]})


def test_an_interaction_names_the_product_of_its_two_columns() -> None:
    interaction = Interaction(left="3_rooms_share", right="ses")
    assert interaction.columns == ("3_rooms_share", "ses")
    # Derived through Product, so the declared name and the fitted one are the
    # same derivation rather than two that could drift.
    assert interaction.name == "3_rooms_share_x_ses"


def test_an_interaction_multiplies_its_two_columns() -> None:
    df = pd.DataFrame({"ses": [1.0, 2.0, 3.0], "rooms": [0.5, 2.0, 4.0]})
    built = Interaction(left="ses", right="rooms").build()
    out = built.set_output(transform="pandas").fit_transform(df)
    assert list(out.columns) == ["ses_x_rooms"]
    np.testing.assert_allclose(out["ses_x_rooms"], [0.5, 4.0, 12.0])


def test_an_interaction_round_trips_with_its_columns_intact() -> None:
    interaction = Interaction(left="3_rooms_share", right="ses")
    assert Interaction.model_validate(interaction.model_dump()) == interaction


def test_an_interaction_step_is_added_only_when_interactions_are_declared() -> None:
    assert [name for name, _ in _interacting_transformer()._build().steps] == [
        "columns"
    ]
    built = _interacting_transformer(
        (Interaction(left="ses", right="3_rooms_share"),)
    )._build()
    assert [name for name, _ in built.steps] == ["columns", "interactions"]


def test_the_interaction_step_passes_the_base_matrix_through() -> None:
    # A ColumnTransformer drops whatever no entry claimed, and an operand is
    # claimed by its product -- so without the base entry the main effects would
    # vanish from the design matrix.
    built = _interacting_transformer(
        (Interaction(left="ses", right="3_rooms_share"),)
    )._build()
    entries = built.named_steps["interactions"].transformers
    assert [name for name, _, _ in entries] == ["base", "ses_x_3_rooms_share"]
    assert entries[1][2] == ["ses", "3_rooms_share"]


def test_cloning_preserves_the_interactions() -> None:
    # One declaration is fit once per fold, so the interactions must survive
    # clone() alongside the plans.
    pairs = (
        Interaction(left="3_rooms_share", right="ses"),
        Interaction(left="3_rooms_share", right="ses_squared"),
    )
    assert clone(_interacting_transformer(pairs)).interactions == pairs


# --- Delegated: interaction mistakes nothing here checks ---------------------
#
# Every one of them is a ColumnTransformer entry whose columns must exist and
# whose output name must be unique, so it refuses all four itself, at fit.


def test_an_interaction_naming_an_unknown_column_is_rejected_when_fitted() -> None:
    # Not checked against the plans' output names: the entry meets the real
    # matrix, and the ColumnTransformer refuses the operand by name.
    with pytest.raises(ValueError, match="no_such_column"):
        _interacting_transformer(
            (Interaction(left="ses", right="no_such_column"),)
        ).fit(_interaction_frame())


def test_the_same_pair_declared_twice_is_rejected_when_fitted() -> None:
    # Two entries, one name. Silently keeping one would leave the caller
    # believing both were added.
    pair = Interaction(left="ses", right="3_rooms_share")
    repeated = (pair, pair)
    with pytest.raises(ValueError, match="not unique"):
        _interacting_transformer(repeated).fit(_interaction_frame())


def test_an_interaction_may_not_shadow_a_base_column() -> None:
    # A product carrying a column's name would replace it; the base entry and
    # the product entry then emit the same name, which is refused.
    transformer = FeatureTransformer(
        plans=(
            _plan("ses_z", "ses", transforms=(Standardize(),)),
            _plan("age", "median_age", transforms=(Standardize(),)),
            _plan("collides", "ses_x_median_age"),
        ),
        interactions=(Interaction(left="ses", right="median_age"),),
    )
    df = pd.DataFrame(
        {"ses": [1.0, 2.0], "median_age": [3.0, 4.0], "ses_x_median_age": [5.0, 6.0]}
    )
    with pytest.raises(ValueError, match="not unique"):
        transformer.fit(df)


def test_a_column_multiplied_by_itself_is_rejected_when_fitted() -> None:
    # The entry would select the same column twice, which pandas refuses. A
    # column times itself is a quadratic term, which Quadratic expresses.
    with pytest.raises(ValueError, match="unique column names"):
        _interacting_transformer((Interaction(left="ses", right="ses"),)).fit(
            _interaction_frame()
        )


# --- sklearn contract --------------------------------------------------------


def test_cloning_preserves_the_declaration() -> None:
    # One declaration is fit once per fold, and clone() is how a fresh unfitted
    # copy is made -- so the parameters must survive it untouched.
    original = _transformer(exposure_column="n_apartments")
    copy = clone(original)
    assert copy.plans == original.plans
    assert copy.exposure_column == "n_apartments"
    assert copy.remainder == original.remainder


def test_build_returns_one_column_transformer_entry_per_plan() -> None:
    transformer = _transformer(
        plans=(
            _plan("ses_z", "ses", transforms=(Standardize(),)),
            _plan("room_share", "3_rooms_share", "4_rooms_share"),
        )
    )
    # _build() returns the whole pipeline; the plans are its "columns" step.
    built = transformer._build().named_steps["columns"]
    assert [name for name, _, _ in built.transformers] == ["ses_z", "room_share"]
    assert built.transformers[1][2] == ["3_rooms_share", "4_rooms_share"]
