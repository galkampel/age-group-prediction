"""Tests for the bank of individual feature transformations.

Each transformation exists to make a malformed specification impossible to
construct, so most of these assert that something raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from age_group_prediction.feature_engineering import (
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

_ADAPTER = TypeAdapter(Transform)

_EVERY_MEMBER = (
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


def test_domain_scale_requires_a_scale() -> None:
    with pytest.raises(ValidationError):
        DomainScale()


def test_domain_scale_rejects_a_zero_or_negative_scale() -> None:
    # Zero would divide by zero; a negative factor would flip the column's sign
    # and silently invert every coefficient read off it.
    with pytest.raises(ValidationError):
        DomainScale(scale=0.0)
    with pytest.raises(ValidationError):
        DomainScale(scale=-1.0)


def test_domain_min_max_requires_both_bounds() -> None:
    with pytest.raises(ValidationError):
        DomainMinMax(minimum=0.0)
    with pytest.raises(ValidationError):
        DomainMinMax(maximum=8.0)


def test_domain_min_max_rejects_an_empty_or_inverted_range() -> None:
    with pytest.raises(ValidationError, match="maximum > minimum"):
        DomainMinMax(minimum=1.0, maximum=0.0)
    with pytest.raises(ValidationError, match="maximum > minimum"):
        DomainMinMax(minimum=4.0, maximum=4.0)


def test_one_hot_requires_categories_and_a_reference() -> None:
    with pytest.raises(ValidationError):
        OneHot()
    with pytest.raises(ValidationError):
        OneHot(categories=("none", "existing"))


def test_one_hot_rejects_a_reference_outside_its_categories() -> None:
    with pytest.raises(ValidationError, match="reference_category"):
        OneHot(categories=("none", "existing"), reference_category="planned")


def test_one_hot_rejects_duplicate_categories() -> None:
    with pytest.raises(ValidationError, match="unique"):
        OneHot(categories=("none", "none"), reference_category="none")


def test_one_hot_orders_the_reference_first_so_drop_first_drops_it() -> None:
    transform = OneHot(
        categories=("existing", "planned", "none"), reference_category="none"
    )
    assert transform.ordered_categories == ("none", "existing", "planned")

    encoder = transform.build()
    assert isinstance(encoder, OneHotEncoder)
    encoded = encoder.fit_transform(np.array([["none"], ["existing"], ["planned"]]))
    names = list(encoder.get_feature_names_out(["school_status"]))
    assert names == ["school_status_existing", "school_status_planned"]
    # The reference row encodes as all-zero: its effect lives in the intercept.
    np.testing.assert_array_equal(encoded[0], [0.0, 0.0])


def test_extra_keys_are_rejected_rather_than_ignored() -> None:
    # A typo in a parameter name must not silently fall back to the default.
    with pytest.raises(ValidationError):
        Standardize(scalar=2)
    with pytest.raises(ValidationError):
        DomainScale(scale=0.1, minimum=0.0)


def test_members_are_frozen() -> None:
    transform = DomainScale(scale=0.1)
    with pytest.raises(ValidationError):
        transform.scale = 0.2


@pytest.mark.parametrize("transform", _EVERY_MEMBER, ids=lambda t: t.kind)
def test_every_member_round_trips_as_its_own_subclass(transform) -> None:
    # The fitted-state bundles serialize specs as JSON, so a reloaded bank has
    # to rebuild the identical transformer, not a look-alike.
    restored = _ADAPTER.validate_python(transform.model_dump())
    assert type(restored) is type(transform)
    assert restored == transform


def test_the_discriminator_names_the_intended_member_in_errors() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _ADAPTER.validate_python({"kind": "domain_min_max", "minimum": 0.0})
    message = str(excinfo.value)
    assert "maximum" in message
    # One error for the tagged member, not one per union member.
    assert excinfo.value.error_count() == 1


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"kind": "sqrt"})


def test_standardize_and_center_differ_in_scaling_only() -> None:
    standardize = Standardize().build()
    center = Center().build()
    assert isinstance(standardize, StandardScaler)
    assert (standardize.with_mean, standardize.with_std) == (True, True)
    assert (center.with_mean, center.with_std) == (True, False)


def test_quadratic_renames_its_output_so_the_linear_term_survives() -> None:
    transformer = Quadratic().build()
    values = np.array([[-2.0], [3.0]])
    np.testing.assert_allclose(transformer.fit_transform(values), [[4.0], [9.0]])
    assert list(transformer.get_feature_names_out(["ses"])) == ["ses_squared"]


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (DomainScale(scale=0.1), [[0.0], [5.0], [10.0]]),
        (DomainMinMax(minimum=0.0, maximum=8.0), [[0.0], [0.0625], [0.125]]),
    ],
    ids=["domain_scale", "domain_min_max"],
)
def test_domain_transforms_use_declared_constants_not_observed_ones(
    transform, expected
) -> None:
    values = np.array([[0.0], [0.5], [1.0]])
    transformer = transform.build()
    np.testing.assert_allclose(transformer.fit_transform(values), expected)
    # Nothing is learned, so a different fold gives the same mapping.
    other = transform.build().fit(np.array([[100.0], [200.0]]))
    np.testing.assert_allclose(other.transform(values), expected)


def test_domain_min_max_does_not_clip_out_of_range_values() -> None:
    # [0, 1] is where the declared range maps, not a guarantee about outputs:
    # a building with more daycares than the declared maximum must stay visible
    # as a value above 1 rather than look like the busiest in-range building.
    transformer = DomainMinMax(minimum=0.0, maximum=8.0).build()
    values = np.array([[-2.0], [0.0], [8.0], [12.0]])
    np.testing.assert_allclose(
        transformer.fit_transform(values), [[-0.25], [0.0], [1.0], [1.5]]
    )


def test_domain_transforms_invert_exactly() -> None:
    values = np.array([[0.0], [0.5], [1.0]])
    for transform in (
        DomainScale(scale=0.1),
        DomainMinMax(minimum=0.0, maximum=8.0),
    ):
        transformer = transform.build()
        np.testing.assert_allclose(
            transformer.inverse_transform(transformer.fit_transform(values)), values
        )


def test_relative_saturation_centers_by_log1p_of_the_mean() -> None:
    # The distinguishing property: the offset is log1p(mean(x)), which stays
    # expressed in the original counts, not mean(log1p(x)).
    counts = np.array([[0.0], [1.0], [8.0]])
    transformer = RelativeSaturation().build()
    transformed = transformer.fit_transform(counts)

    expected = np.log1p(counts) - np.log1p(counts.mean())
    np.testing.assert_allclose(transformed, expected)

    # ... and that is genuinely not what a Log1p -> Center chain produces.
    chained = StandardScaler(with_mean=True, with_std=False).fit_transform(
        np.log1p(counts)
    )
    assert not np.allclose(transformed, chained)


def test_relative_saturation_uses_the_fit_fold_mean_on_later_folds() -> None:
    fit_counts = np.array([[0.0], [1.0], [8.0]])
    transformer = RelativeSaturation().build().fit(fit_counts)
    eval_counts = np.array([[2.0], [4.0]])
    np.testing.assert_allclose(
        transformer.transform(eval_counts),
        np.log1p(eval_counts) - np.log1p(fit_counts.mean()),
    )


def test_relative_saturation_inverts_and_names_its_output() -> None:
    counts = np.array([[0.0], [1.0], [8.0]])
    transformer = RelativeSaturation().build()
    np.testing.assert_allclose(
        transformer.inverse_transform(transformer.fit_transform(counts)), counts
    )
    assert list(transformer.get_feature_names_out(["n_daycares_500m"])) == [
        "n_daycares_500m_sat"
    ]


def test_relative_saturation_rejects_values_at_or_below_minus_one() -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        RelativeSaturation().build().fit(np.array([[-1.0], [2.0]]))


# --- scikit-learn estimator contract for the fitted scaler -------------------
#
# The scaler is the one member that learns from the fit fold, so it is the one
# member that can leak a fold's state. These pin the contract that prevents it.


def _daycare_frame(
    values: list[float], column: str = "n_daycares_500m"
) -> pd.DataFrame:
    return pd.DataFrame({column: values})


def test_relative_saturation_raises_not_fitted_before_fit() -> None:
    # Pipeline and ColumnTransformer key their error handling on NotFittedError;
    # a bare AttributeError on a missing mean_ would read as a bug in sklearn.
    transformer = RelativeSaturation().build()
    with pytest.raises(NotFittedError):
        transformer.transform(_daycare_frame([1.0, 2.0]))
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


def test_relative_saturation_records_the_fit_frames_columns() -> None:
    transformer = RelativeSaturation().build().fit(_daycare_frame([0.0, 1.0, 8.0]))
    assert transformer.n_features_in_ == 1
    assert list(transformer.feature_names_in_) == ["n_daycares_500m"]
    # Recorded names are what let get_feature_names_out answer with no argument.
    assert list(transformer.get_feature_names_out()) == ["n_daycares_500m_sat"]


def test_relative_saturation_rejects_a_later_frame_whose_columns_moved() -> None:
    # The failure this guards: a later fold arriving with different or
    # reordered columns would otherwise be centered by a mean belonging to some
    # other variable, silently.
    transformer = RelativeSaturation().build().fit(_daycare_frame([0.0, 1.0, 8.0]))

    with pytest.raises(ValueError):
        transformer.transform(_daycare_frame([1.0], column="median_age"))
    with pytest.raises(ValueError):
        transformer.transform(
            pd.DataFrame({"n_daycares_500m": [1.0], "median_age": [30.0]})
        )


def test_cloning_a_fitted_saturation_scaler_yields_an_unfitted_one() -> None:
    # clone() is how sklearn gives each CV fold a fresh estimator. If learned
    # state survived it, every fold after the first would be fitted on leaked
    # statistics.
    fitted = RelativeSaturation().build().fit(_daycare_frame([0.0, 1.0, 8.0]))
    fresh = clone(fitted)
    assert not hasattr(fresh, "mean_")
    with pytest.raises(NotFittedError):
        fresh.transform(_daycare_frame([1.0]))


def test_relative_saturation_returns_a_named_frame_under_pandas_output() -> None:
    # Column identity travels through get_feature_names_out, and sklearn
    # rebuilds the df from it; the transformer itself stays array-based.
    df = _daycare_frame([0.0, 1.0, 8.0])
    transformer = RelativeSaturation().build().set_output(transform="pandas")
    transformed = transformer.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == ["n_daycares_500m_sat"]
    assert transformed.index.equals(df.index)
