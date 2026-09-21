"""Tests for applying a set of column plans to data.

Expected values are written out by hand rather than recomputed from the
implementation, so a test fails when the arithmetic changes rather than
following it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

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
    RelativeSaturation,
    Standardize,
)


def _plan(name: str, *columns: str, transforms=()) -> ColumnPlan:
    return ColumnPlan(name=name, columns=columns, transforms=transforms)


def _frame() -> pd.DataFrame:
    """A small df with hand-chosen means, on a non-default index."""
    return pd.DataFrame(
        {
            # mean 0.2, so centering gives -0.1, 0.0, 0.1
            "3_rooms_share": [0.1, 0.2, 0.3],
            "4_rooms_share": [0.2, 0.3, 0.4],
            "n_daycares_500m": [0.0, 4.0, 8.0],
            "ses": [1.0, 2.0, 3.0],
            "school_status": ["none", "existing", "planned"],
            "unplanned": [7.0, 8.0, 9.0],
        },
        index=[10, 11, 12],
    )


def _tf(**overrides) -> FeatureTransformer:
    defaults = {
        "plans": (
            _plan(
                "room_share",
                "3_rooms_share",
                "4_rooms_share",
                transforms=(Center(), DomainScale(scale=0.1)),
            ),
            _plan(
                "daycare",
                "n_daycares_500m",
                transforms=(DomainMinMax(minimum=0.0, maximum=8.0),),
            ),
            _plan(
                "school",
                "school_status",
                transforms=(
                    OneHot(
                        categories=("none", "existing", "planned"),
                        reference_category="none",
                    ),
                ),
            ),
        ),
    }
    return FeatureTransformer(**{**defaults, **overrides})


# --- Values ------------------------------------------------------------------


def test_room_shares_are_centered_then_scaled_to_ten_point_units() -> None:
    out = _tf().fit_transform(_frame())
    # (x - 0.2) / 0.1 and (x - 0.3) / 0.1: a coefficient per 10 points of share.
    np.testing.assert_allclose(out["3_rooms_share"], [-1.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out["4_rooms_share"], [-1.0, 0.0, 1.0], atol=1e-12)


def test_daycare_counts_map_onto_the_declared_domain() -> None:
    out = _tf().fit_transform(_frame())
    # (x - 0) / (8 - 0), so the declared maximum lands on 1.
    np.testing.assert_allclose(out["n_daycares_500m"], [0.0, 0.5, 1.0], atol=1e-12)


def test_the_one_hot_drops_the_reference_level() -> None:
    out = _tf().fit_transform(_frame())
    assert "school_status_none" not in out.columns
    np.testing.assert_allclose(out["school_status_existing"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out["school_status_planned"], [0.0, 0.0, 1.0])


def test_a_relative_saturation_centers_on_log1p_of_the_mean() -> None:
    # log1p(x) - log1p(mean), not the mean of log1p: the reference stays
    # expressed in the original counts, so the output is not mean-zero.
    df = pd.DataFrame({"n_daycares_500m": [0.0, 4.0, 8.0]})
    transformer = FeatureTransformer(
        plans=(
            _plan("daycare", "n_daycares_500m", transforms=(RelativeSaturation(),)),
        ),
    )
    out = transformer.fit_transform(df)
    expected = np.log1p([0.0, 4.0, 8.0]) - np.log1p(4.0)
    np.testing.assert_allclose(out["n_daycares_500m_sat"], expected, atol=1e-12)


# --- Structure ---------------------------------------------------------------


def test_unplanned_columns_are_dropped_by_default() -> None:
    out = _tf().fit_transform(_frame())
    assert "unplanned" not in out.columns
    assert list(out.columns) == [
        "3_rooms_share",
        "4_rooms_share",
        "n_daycares_500m",
        "school_status_existing",
        "school_status_planned",
    ]


def test_a_passthrough_remainder_keeps_unplanned_columns_untouched() -> None:
    transformer = FeatureTransformer(
        plans=(_plan("ses_z", "ses", transforms=(Standardize(),)),),
        remainder="passthrough",
    )
    out = transformer.fit_transform(_frame())
    assert "unplanned" in out.columns
    np.testing.assert_allclose(out["unplanned"], [7.0, 8.0, 9.0])


def test_feature_names_out_matches_the_transformed_frame() -> None:
    transformer = _tf().fit(_frame())
    out = transformer.transform(_frame())
    assert list(transformer.get_feature_names_out()) == list(out.columns)


def test_transform_preserves_the_frame_index() -> None:
    out = _tf().fit_transform(_frame())
    assert list(out.index) == [10, 11, 12]


# --- Interactions ------------------------------------------------------------
#
# Products of the *transformed* columns, computed in a second ColumnTransformer
# over the base matrix, so they inherit whatever centering and scaling their
# operands were given.


def _room_by_daycare():
    return (
        Interaction(left="3_rooms_share", right="n_daycares_500m"),
        Interaction(left="4_rooms_share", right="n_daycares_500m"),
    )


def test_no_interactions_yields_no_product_columns() -> None:
    out = _tf().fit_transform(_frame())
    assert [name for name in out.columns if "_x_" in name] == []


def test_a_cross_yields_one_column_per_pair() -> None:
    out = _tf(interactions=_room_by_daycare()).fit_transform(_frame())
    assert [name for name in out.columns if "_x_" in name] == [
        "3_rooms_share_x_n_daycares_500m",
        "4_rooms_share_x_n_daycares_500m",
    ]
    # Never a within-group pair: the room shares are not multiplied together.
    assert "3_rooms_share_x_4_rooms_share" not in out.columns


def test_an_interaction_multiplies_the_transformed_columns_not_the_raw_ones() -> None:
    out = _tf(interactions=_room_by_daycare()).fit_transform(_frame())
    # Transformed operands: [-1, 0, 1] and [0, 0.5, 1], so the product is
    # [0, 0, 1]. The raw product would be 0.1*0, 0.2*4, 0.3*8 = [0, 0.8, 2.4].
    np.testing.assert_allclose(
        out["3_rooms_share_x_n_daycares_500m"], [0.0, 0.0, 1.0], atol=1e-12
    )
    np.testing.assert_allclose(
        out["4_rooms_share_x_n_daycares_500m"], [0.0, 0.0, 1.0], atol=1e-12
    )


def test_the_base_matrix_survives_the_interaction_step_in_order() -> None:
    out = _tf(interactions=_room_by_daycare()).fit_transform(_frame())
    assert list(out.columns) == [
        "3_rooms_share",
        "4_rooms_share",
        "n_daycares_500m",
        "school_status_existing",
        "school_status_planned",
        "3_rooms_share_x_n_daycares_500m",
        "4_rooms_share_x_n_daycares_500m",
    ]


def test_feature_names_out_includes_the_interaction_columns() -> None:
    transformer = _tf(interactions=_room_by_daycare()).fit(_frame())
    out = transformer.transform(_frame())
    assert list(transformer.get_feature_names_out()) == list(out.columns)


def test_an_interaction_may_name_a_one_hot_level() -> None:
    # One level of the encoding, so the product reads as that level's own slope.
    interactions = (
        Interaction(left="n_daycares_500m", right="school_status_existing"),
        Interaction(left="n_daycares_500m", right="school_status_planned"),
    )
    out = _tf(interactions=interactions).fit_transform(_frame())
    # [0, 0.5, 1] times [0, 1, 0] and [0, 0, 1].
    np.testing.assert_allclose(
        out["n_daycares_500m_x_school_status_existing"], [0.0, 0.5, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        out["n_daycares_500m_x_school_status_planned"], [0.0, 0.0, 1.0], atol=1e-12
    )


def test_an_interaction_may_name_a_passthrough_remainder_column() -> None:
    # The step meets the real matrix rather than the plans' declared output
    # names, so a column the remainder carried through is a legal operand.
    transformer = FeatureTransformer(
        plans=(_plan("ses_z", "ses", transforms=(Standardize(),)),),
        interactions=(Interaction(left="ses", right="unplanned"),),
        remainder="passthrough",
    )
    out = transformer.fit_transform(_frame())
    np.testing.assert_allclose(
        out["ses_x_unplanned"], out["ses"] * [7.0, 8.0, 9.0], atol=1e-12
    )


def test_a_later_fold_multiplies_columns_built_from_the_fit_folds_statistics() -> None:
    fit_frame = _frame()
    transformer = _tf(interactions=_room_by_daycare()).fit(fit_frame)
    later = fit_frame.copy()
    later["3_rooms_share"] = [0.5, 0.6, 0.7]
    out = transformer.transform(later)
    # Centered by the fit df's mean of 0.2, giving [3, 4, 5], times the
    # daycare column's [0, 0.5, 1]. Recentering on the later df would give
    # [-1, 0, 1] and so a product of [0, 0, 1].
    np.testing.assert_allclose(
        out["3_rooms_share_x_n_daycares_500m"], [0.0, 2.0, 5.0], atol=1e-12
    )


# --- Fold discipline ---------------------------------------------------------


def test_a_later_frame_is_centered_by_the_fit_frames_means() -> None:
    # The point of fitting: a validation fold must not be recentered on itself,
    # which would leak its own distribution into the design matrix.
    fit_frame = _frame()
    transformer = _tf().fit(fit_frame)
    later = fit_frame.copy()
    later["3_rooms_share"] = [0.5, 0.6, 0.7]
    out = transformer.transform(later)
    # Centered by the fit df's mean of 0.2, not the later df's 0.6.
    np.testing.assert_allclose(out["3_rooms_share"], [3.0, 4.0, 5.0], atol=1e-12)


# --- Failures ----------------------------------------------------------------


def test_transforming_before_fitting_raises() -> None:
    with pytest.raises(NotFittedError):
        _tf().transform(_frame())


def test_a_missing_planned_column_is_rejected_when_fitted() -> None:
    # ColumnTransformer names the column itself, one at a time.
    df = _frame().drop(columns=["n_daycares_500m"])
    with pytest.raises(ValueError, match="n_daycares_500m"):
        _tf().fit(df)


@pytest.mark.filterwarnings("ignore:.*encountered in log:RuntimeWarning")
def test_a_log_over_a_zero_is_caught_by_the_output_check() -> None:
    # numpy returns -inf without raising, so the poisoned column is what is
    # detected rather than the input that caused it.
    transformer = FeatureTransformer(
        plans=(_plan("daycare", "n_daycares_500m", transforms=(Log(),)),),
    )
    with pytest.raises(ValueError, match="non-finite") as excinfo:
        transformer.fit_transform(_frame())
    assert "n_daycares_500m" in str(excinfo.value)


@pytest.mark.filterwarnings("ignore:.*encountered in log:RuntimeWarning")
def test_a_log1p_at_minus_one_is_caught_by_the_output_check() -> None:
    df = pd.DataFrame({"balance": [-1.0, 0.0, 1.0]})
    transformer = FeatureTransformer(
        plans=(_plan("balance", "balance", transforms=(Log1p(),)),),
    )
    with pytest.raises(ValueError, match="non-finite"):
        transformer.fit_transform(df)


def test_a_relative_saturation_below_minus_one_is_rejected_at_fit() -> None:
    # This one still fails during fit: the scaler validates its own input.
    df = pd.DataFrame({"balance": [-2.0, 0.0, 1.0]})
    transformer = FeatureTransformer(
        plans=(_plan("balance", "balance", transforms=(RelativeSaturation(),)),),
    )
    with pytest.raises(ValueError, match="greater than -1"):
        transformer.fit(df)


@pytest.mark.filterwarnings("ignore:.*encountered in log:RuntimeWarning")
def test_a_later_frame_that_poisons_the_matrix_is_rejected_too() -> None:
    # The fit df is fine; the one being transformed is not.
    transformer = FeatureTransformer(
        plans=(_plan("count", "count", transforms=(Log(),)),),
    ).fit(pd.DataFrame({"count": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="non-finite"):
        transformer.transform(pd.DataFrame({"count": [1.0, 0.0, 3.0]}))


# --- The exposure offset -----------------------------------------------------


def _exposure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"ses": [1.0, 2.0, 3.0], "n_apartments": [1.0, 10.0, 100.0]},
        index=[10, 11, 12],
    )


def _exposure_tf(**overrides) -> FeatureTransformer:
    defaults = {
        "plans": (_plan("ses_z", "ses", transforms=(Standardize(),)),),
        "exposure_column": "n_apartments",
    }
    return FeatureTransformer(**{**defaults, **overrides})


def test_the_offset_is_the_log_of_the_exposure_column() -> None:
    offset = _exposure_tf().log_exposure(_exposure_frame())
    assert offset.name == "log_n_apartments"
    # log(1), log(10), log(100) -- hand-checked against the natural log.
    np.testing.assert_allclose(offset.to_numpy(), [0.0, np.log(10.0), np.log(100.0)])
    assert list(offset.index) == [10, 11, 12]


def test_there_is_no_offset_when_no_exposure_is_declared() -> None:
    assert _tf().log_exposure(_frame()) is None


def test_an_unplanned_exposure_is_dropped_from_the_design_matrix() -> None:
    # Not a guarantee about exposures -- it follows from no plan claiming the
    # column and the remainder dropping it. A plan may claim it (see
    # test_an_exposure_may_also_be_a_predictor).
    transformer = _exposure_tf().fit(_exposure_frame())
    names = list(transformer.get_feature_names_out())
    assert "n_apartments" not in names
    assert "log_n_apartments" not in names
    assert names == ["ses"]


def test_a_non_positive_exposure_is_rejected_when_the_offset_is_asked_for() -> None:
    df = _exposure_frame()
    df.loc[11, "n_apartments"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _exposure_tf().log_exposure(df)


def test_a_missing_or_blank_exposure_column_raises_when_the_offset_is_asked_for() -> (
    None
):
    # Not checked at fit: the exposure is not part of the design matrix, so
    # nothing touches it until the offset is requested. pandas names it.
    with pytest.raises(KeyError, match="n_apartments"):
        _exposure_tf().log_exposure(_exposure_frame().drop(columns=["n_apartments"]))
    # A blank name must not be silently treated as "no exposure".
    with pytest.raises(KeyError):
        _exposure_tf(exposure_column="").log_exposure(_exposure_frame())
