"""Tests for the fit-free, row-wise preprocessing steps.

Expected values are written out by hand, so a test fails when the arithmetic
changes rather than following it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from age_group_prediction.preprocessing import ShareTransformer

ROOMS = ("3_rooms", "4_rooms", "5_rooms", "6_rooms")


def _frame() -> pd.DataFrame:
    """Room counts summing to ten per row, on a non-default index.

    ``n_apartments`` disagrees with that sum in the last row, so a test can
    tell the two candidate denominators apart.
    """
    return pd.DataFrame(
        {
            "building_id": ["b1", "b2", "b3"],
            "ses": [0.1, 0.5, 0.9],
            "3_rooms": [1, 2, 5],
            "4_rooms": [2, 3, 2],
            "5_rooms": [3, 4, 2],
            "6_rooms": [4, 1, 1],
            "n_apartments": [10, 10, 99],
            "school_status": ["none", "existing", "planned"],
        },
        index=[10, 11, 12],
    )


# --- Values -----------------------------------------------------------


def test_each_share_is_its_count_over_the_row_sum_of_the_declared_columns() -> None:
    transformed = ShareTransformer(ROOMS, reference_column="6_rooms").fit_transform(
        _frame()
    )

    np.testing.assert_allclose(transformed["3_rooms_share"], [0.1, 0.2, 0.5])
    np.testing.assert_allclose(transformed["4_rooms_share"], [0.2, 0.3, 0.2])
    np.testing.assert_allclose(transformed["5_rooms_share"], [0.3, 0.4, 0.2])


def test_the_shares_sum_to_one_when_no_reference_is_dropped() -> None:
    transformed = ShareTransformer(ROOMS).fit_transform(_frame())

    shares = [f"{name}_share" for name in ROOMS]
    np.testing.assert_allclose(transformed[shares].sum(axis=1), [1.0, 1.0, 1.0])


def test_the_denominator_is_the_row_sum_not_an_outside_apartment_total() -> None:
    # Row b3 has ten rooms but n_apartments of 99: 5/10, not 5/99.
    transformed = ShareTransformer(ROOMS, reference_column="6_rooms").fit_transform(
        _frame()
    )

    assert transformed.loc[12, "3_rooms_share"] == pytest.approx(0.5)


def test_the_reference_share_is_not_emitted_and_its_count_is_dropped() -> None:
    transformed = ShareTransformer(ROOMS, reference_column="6_rooms").fit_transform(
        _frame()
    )

    assert "6_rooms_share" not in transformed.columns
    assert "6_rooms" not in transformed.columns
    # The 6-room counts are 4, 1 and 1 out of ten, so the rest sum to the rest.
    shares = ["3_rooms_share", "4_rooms_share", "5_rooms_share"]
    np.testing.assert_allclose(transformed[shares].sum(axis=1), [0.6, 0.9, 0.9])


def test_each_row_is_transformed_independently_of_the_others() -> None:
    # Why running before the split cannot leak.
    forwards = ShareTransformer(ROOMS).fit_transform(_frame())
    backwards = ShareTransformer(ROOMS).fit_transform(_frame().iloc[::-1])

    pd.testing.assert_frame_equal(forwards, backwards.loc[forwards.index])


# --- Structure --------------------------------------------------------


def test_columns_outside_the_share_group_pass_through_untouched() -> None:
    frame = _frame()
    transformed = ShareTransformer(ROOMS).fit_transform(frame)

    for column in ("building_id", "ses", "n_apartments", "school_status"):
        pd.testing.assert_series_equal(transformed[column], frame[column])


def test_passthrough_dtypes_are_unchanged() -> None:
    # column_schema_hash hashes dtypes and gates the lockbox manifest, so a
    # coerced int would break split replay.
    frame = _frame()
    transformed = ShareTransformer(ROOMS).fit_transform(frame)

    assert transformed["n_apartments"].dtype == frame["n_apartments"].dtype
    assert transformed["building_id"].dtype == frame["building_id"].dtype
    assert transformed["school_status"].dtype == frame["school_status"].dtype


def test_the_shares_are_appended_after_the_columns_that_survived() -> None:
    transformed = ShareTransformer(ROOMS, reference_column="6_rooms").fit_transform(
        _frame()
    )

    assert list(transformed.columns) == [
        "building_id",
        "ses",
        "n_apartments",
        "school_status",
        "3_rooms_share",
        "4_rooms_share",
        "5_rooms_share",
    ]


def test_transform_preserves_the_frame_index() -> None:
    transformed = ShareTransformer(ROOMS).fit_transform(_frame())

    assert list(transformed.index) == [10, 11, 12]


def test_feature_names_out_matches_the_transformed_frame() -> None:
    # set_output(transform="pandas") relabels using this method, so a
    # disagreement would rename columns silently.
    transformer = ShareTransformer(ROOMS, reference_column="6_rooms").fit(_frame())

    assert list(transformer.get_feature_names_out()) == list(
        transformer.transform(_frame()).columns
    )


def test_drop_inputs_false_keeps_the_counts_beside_their_shares() -> None:
    transformed = ShareTransformer(ROOMS, drop_inputs=False).fit_transform(_frame())

    np.testing.assert_allclose(transformed["3_rooms"], [1, 2, 5])
    np.testing.assert_allclose(transformed["3_rooms_share"], [0.1, 0.2, 0.5])


def test_a_suffix_other_than_the_default_names_every_share() -> None:
    transformed = ShareTransformer(ROOMS, suffix="_frac").fit_transform(_frame())

    assert "3_rooms_frac" in transformed.columns
    assert "3_rooms_share" not in transformed.columns


def test_there_is_no_inverse_transform_because_the_row_sum_is_lost() -> None:
    # A share vector maps to an infinite family of count vectors.
    assert not hasattr(ShareTransformer, "inverse_transform")


# --- scikit-learn contract --------------------------------------------


def test_a_clone_is_unfitted_and_reproduces_the_same_output() -> None:
    fitted = ShareTransformer(ROOMS, reference_column="6_rooms").fit(_frame())
    fresh = clone(fitted)

    assert fresh.get_params() == fitted.get_params()
    with pytest.raises(NotFittedError):
        fresh.transform(_frame())
    pd.testing.assert_frame_equal(
        fresh.fit_transform(_frame()), fitted.transform(_frame())
    )


def test_transforming_before_fitting_raises() -> None:
    with pytest.raises(NotFittedError):
        ShareTransformer(ROOMS).transform(_frame())


def test_get_feature_names_out_before_fitting_raises() -> None:
    with pytest.raises(NotFittedError):
        ShareTransformer(ROOMS).get_feature_names_out()


# --- Failures ---------------------------------------------------------


def test_a_row_whose_share_columns_sum_to_zero_is_rejected() -> None:
    # The one check the transformer makes itself.
    frame = _frame()
    frame.loc[11, list(ROOMS)] = 0

    with pytest.raises(ValueError, match="sum to zero") as excinfo:
        ShareTransformer(ROOMS).fit_transform(frame)
    # The label, not the position: after a split the index is not 0..n.
    assert "11" in str(excinfo.value)


def test_a_missing_share_column_raises_a_key_error() -> None:
    # Left to pandas, which already names the column.
    frame = _frame().drop(columns=["3_rooms"])

    with pytest.raises(KeyError, match="3_rooms"):
        ShareTransformer(ROOMS).fit_transform(frame)


def test_a_non_numeric_share_column_raises_a_value_error() -> None:
    # Left to numpy, which already names the offending value.
    frame = _frame()
    frame["3_rooms"] = ["a", "b", "c"]

    with pytest.raises(ValueError, match="could not convert"):
        ShareTransformer(ROOMS).fit_transform(frame)
