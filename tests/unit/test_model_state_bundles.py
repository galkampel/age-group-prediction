"""One state-bundle contract, checked for every model family.

A state bundle (``age_group_prediction.state_bundle``) is the fitted-model
artifact that Gate 7 logs to MLflow. Plan section 11 requires it to be
reconstructable without opaque pickling, so every family is held to the same
checks:

* a JSON round trip reloads **without the training frame and without refitting**
  and reproduces point predictions, pointwise log probabilities, parametric
  distributions, and model metadata exactly;
* predictive draws reproduce exactly, and models whose draws refit on the
  training frame (the bootstrap models) require that frame;
* a different training frame and an incompatible bundle are refused by name;
* the bundle holds no training rows, and editing it cannot alter the model.

Adding a model family means adding one ``_Case`` to ``_CASES``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    BaseAgeGroupModel,
    BayesianConditionalModel,
    DirectCohortModel,
    IndependentTotalProbabilityModel,
    PredictionConfig,
)
from age_group_prediction.fitted_features import FittedFeatureTransformer
from age_group_prediction.results import PredictionResult
from tests.unit.test_bayesian_conditional import _informative_posterior_stub
from tests.unit.test_direct_cohort import _make_config as _direct_config
from tests.unit.test_independent_total_probability import (
    _config as _independent_config,
)

# Not equal to any posterior sample count or trial count used below, so the
# "no array has one entry per building" check cannot match by coincidence.
_ROW_COUNT = 36
_FLOAT_COVARIATES = (
    "ses",
    "avg_household_size",
    "median_age",
    "3_rooms_share",
    "4_rooms_share",
    "5_rooms_share",
)


@pytest.fixture(scope="module")
def modeling_df() -> pd.DataFrame:
    """Return a small frame every family can fit.

    Neighborhood IDs are numeric to exercise the Bayesian lookup's JSON
    conversion. Every building has a child in every cohort, so no bootstrap
    resample leaves the composition model unidentified.
    """
    rng = np.random.default_rng(11)
    counts = rng.integers(1, 5, size=(_ROW_COUNT, 3))
    frame = pd.DataFrame(
        {
            # Distinctive IDs, so a leaked ID is findable in the serialized text.
            "building_id": 900_001 + np.arange(_ROW_COUNT),
            "neighborhood_id": np.repeat(np.arange(3), _ROW_COUNT // 3),
            "ses": rng.normal(size=_ROW_COUNT),
            "avg_household_size": rng.uniform(1.8, 3.8, size=_ROW_COUNT),
            "median_age": rng.uniform(24.0, 58.0, size=_ROW_COUNT),
            "n_daycares_500m": rng.integers(0, 6, size=_ROW_COUNT),
            "n_apartments": rng.integers(8, 60, size=_ROW_COUNT),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=_ROW_COUNT),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=_ROW_COUNT),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=_ROW_COUNT),
            "school_status": rng.choice(["none", "existing", "planned"], size=_ROW_COUNT),
            "n_kindergarten": counts[:, 0],
            "n_elementary": counts[:, 1],
            "n_highschool": counts[:, 2],
        }
    )
    frame["n_children_total"] = counts.sum(axis=1)
    return frame


@dataclass(frozen=True)
class _Case:
    """How to fit one model family, and whether its draws refit on training data."""

    name: str
    fit: Callable[[pd.DataFrame, pytest.MonkeyPatch], BaseAgeGroupModel]
    draws_need_training_frame: bool


def _fit_direct(family: str) -> Callable[[pd.DataFrame, pytest.MonkeyPatch], BaseAgeGroupModel]:
    """Return a fitter for a ``DirectCohortModel`` of the given ``family``."""
    def fit(frame: pd.DataFrame, patch: pytest.MonkeyPatch) -> BaseAgeGroupModel:
        """Fit the tree model on ``frame``; ``patch`` is unused."""
        del patch
        return DirectCohortModel(direct_cohort_config=_direct_config(family=family)).fit(
            frame, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(3)
        )

    return fit


def _fit_independent(
    family: str,
) -> Callable[[pd.DataFrame, pytest.MonkeyPatch], BaseAgeGroupModel]:
    """Return a fitter for the independent model with ``family`` totals."""
    def fit(frame: pd.DataFrame, patch: pytest.MonkeyPatch) -> BaseAgeGroupModel:
        """Fit the independent model on ``frame``; ``patch`` is unused."""
        del patch
        return IndependentTotalProbabilityModel(
            independent_config=_independent_config(total_family=family)  # type: ignore[arg-type]
        ).fit(frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(3))

    return fit


def _fit_bayesian(frame: pd.DataFrame, patch: pytest.MonkeyPatch) -> BaseAgeGroupModel:
    """Fit the Bayesian model on ``frame`` with NUTS stubbed through ``patch``."""
    # Stubbed NUTS: a bundle stores whatever posterior the fit produced, so a
    # real sampler adds runtime without adding coverage here.
    _informative_posterior_stub(patch)
    return BayesianConditionalModel().fit(
        frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(3)
    )


_CASES = (
    _Case("direct-poisson", _fit_direct("poisson"), draws_need_training_frame=True),
    _Case("direct-normal", _fit_direct("normal"), draws_need_training_frame=True),
    _Case("direct-nb2", _fit_direct("nb2"), draws_need_training_frame=True),
    _Case("independent-nb2", _fit_independent("nb2"), draws_need_training_frame=True),
    _Case(
        "independent-poisson", _fit_independent("poisson"), draws_need_training_frame=True
    ),
    _Case("bayesian", _fit_bayesian, draws_need_training_frame=False),
)


@pytest.fixture(scope="module", params=_CASES, ids=[case.name for case in _CASES])
def fitted(
    request: pytest.FixtureRequest, modeling_df: pd.DataFrame
) -> tuple[_Case, BaseAgeGroupModel]:
    """Return each case with its model, fitted once per module."""
    case: _Case = request.param
    with pytest.MonkeyPatch.context() as patch:
        return case, case.fit(modeling_df, patch)


def _round_trip(model: BaseAgeGroupModel) -> dict[str, object]:
    """Export a bundle through real JSON text, as an artifact store would."""
    return json.loads(json.dumps(model.to_state_bundle()))


def _json_normalized(value: object) -> object:
    """Compare structures by content; JSON turns tuples into lists."""
    return json.loads(json.dumps(value))


def _fitted_metadata(model: BaseAgeGroupModel) -> dict[str, object]:
    """Return the metadata that describes the fit, which a bundle must restore.

    Excluded, because they record calls rather than fitted state and are not
    part of a bundle: durations, derived seeds, and counters set by the most
    recent ``predict`` (bootstrap replicate counts, unseen-neighborhood rows).
    """
    metadata = _json_normalized(model.get_metadata())
    assert isinstance(metadata, dict)
    model_section = metadata["model"]
    model_section.pop("last_prediction_unseen_neighborhood_count", None)
    for counter in ("bootstrap_successful_replicates", "bootstrap_failed_replicates"):
        model_section.get("diagnostics", {}).pop(counter, None)
    return {
        key: metadata[key]
        for key in (
            "model",
            "feature_spec",
            "preprocessing",
            "training_data_hash",
            "training_schema_hash",
        )
    }


def _assert_same_predictions(restored: PredictionResult, original: PredictionResult) -> None:
    """Assert two prediction results agree exactly, field by field."""
    np.testing.assert_array_equal(restored.total_mean, original.total_mean)
    np.testing.assert_array_equal(restored.cohort_means, original.cohort_means)
    # Result mappings are read-only proxies, which `np.testing.assert_equal`
    # does not recurse into, so compare them key by key.
    for field in (
        "pointwise_log_probabilities",
        "predictive_draws",
        "prediction_intervals",
    ):
        actual, expected = getattr(restored, field), getattr(original, field)
        assert (actual is None) == (expected is None), field
        assert set(actual or {}) == set(expected or {}), field
        for name, values in (expected or {}).items():
            np.testing.assert_array_equal(actual[name], values, err_msg=f"{field}[{name}]")
    # Specs carry array-valued parameters, so dataclass equality is ambiguous.
    restored_specs = dict(restored.parametric_distributions or {})
    original_specs = dict(original.parametric_distributions or {})
    assert set(restored_specs) == set(original_specs)
    for name, spec in original_specs.items():
        assert restored_specs[name].family == spec.family, name
        for parameter in ("dispersion", "scale"):
            np.testing.assert_array_equal(
                getattr(restored_specs[name], parameter),
                getattr(spec, parameter),
                err_msg=f"parametric_distributions[{name}].{parameter}",
            )
    assert (
        restored.pointwise_log_probability_scope
        == original.pointwise_log_probability_scope
    )


def test_bundle_reloads_without_training_data_or_refitting(
    fitted: tuple[_Case, BaseAgeGroupModel],
    modeling_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without refitting, a bundle alone reproduces predictions and fitted metadata."""
    _, model = fitted
    bundle = _round_trip(model)

    def refuse(*args: object, **kwargs: object) -> None:
        """Fail the test if loading calls any fit method."""
        raise AssertionError("loading a state bundle must not fit anything")

    monkeypatch.setattr(BaseAgeGroupModel, "fit", refuse)
    monkeypatch.setattr(FittedFeatureTransformer, "fit", refuse)

    reloaded = type(model).from_state_bundle(bundle)

    assert reloaded.is_fitted
    config = PredictionConfig(include_pointwise_log_probabilities=True)
    _assert_same_predictions(
        reloaded.predict(modeling_df, prediction_config=config, rng=np.random.default_rng(8)),
        model.predict(modeling_df, prediction_config=config, rng=np.random.default_rng(8)),
    )
    assert _fitted_metadata(reloaded) == _fitted_metadata(model)


@pytest.mark.parametrize("form", ["python objects", "sorted-key JSON"])
def test_bundle_reloads_from_any_faithful_serialization(
    fitted: tuple[_Case, BaseAgeGroupModel], modeling_df: pd.DataFrame, form: str
) -> None:
    """Neither skipping JSON nor a writer that reorders keys may change the model."""
    _, model = fitted
    bundle = model.to_state_bundle()
    if form == "sorted-key JSON":
        bundle = json.loads(json.dumps(bundle, sort_keys=True))

    reloaded = type(model).from_state_bundle(bundle)

    _assert_same_predictions(reloaded.predict(modeling_df), model.predict(modeling_df))


def test_predictive_draws_match_and_require_the_training_frame_only_to_refit(
    fitted: tuple[_Case, BaseAgeGroupModel], modeling_df: pd.DataFrame
) -> None:
    """Draws reproduce exactly; bootstrap models need ``train_df`` to produce them."""
    case, model = fitted
    bundle = _round_trip(model)
    config = PredictionConfig(n_predictive_draws=4, interval_levels=(0.8,))
    eval_df = modeling_df.iloc[:6]
    expected = model.predict(eval_df, prediction_config=config, rng=np.random.default_rng(9))

    reloaded = type(model).from_state_bundle(bundle)
    if case.draws_need_training_frame:
        with pytest.raises(RuntimeError, match="train_df"):
            reloaded.predict(eval_df, prediction_config=config, rng=np.random.default_rng(9))
        reloaded = type(model).from_state_bundle(bundle, train_df=modeling_df)

    _assert_same_predictions(
        reloaded.predict(eval_df, prediction_config=config, rng=np.random.default_rng(9)),
        expected,
    )


@pytest.mark.parametrize("change", ["row values", "column dtype"])
def test_loader_rejects_a_different_training_frame(
    fitted: tuple[_Case, BaseAgeGroupModel], modeling_df: pd.DataFrame, change: str
) -> None:
    """Uncertainty must never be computed from a partition the model did not see."""
    _, model = fitted
    changed = modeling_df.copy()
    if change == "row values":
        changed.loc[changed.index[0], "ses"] += 5.0
    else:
        changed["n_apartments"] = changed["n_apartments"].astype(float)

    with pytest.raises(ValueError, match="train_df does not match the bundle"):
        type(model).from_state_bundle(_round_trip(model), train_df=changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("bundle_format", "99", "bundle format", id="future-format"),
        pytest.param("model_class", "SomeOtherModel", "written by", id="wrong-model"),
        pytest.param(
            "implementation_version", "0", "implementation version", id="stale-version"
        ),
    ],
)
def test_loader_refuses_an_incompatible_bundle(
    fitted: tuple[_Case, BaseAgeGroupModel], field: str, value: str, message: str
) -> None:
    """A wrong format, model class, or implementation version is refused by name."""
    _, model = fitted
    bundle = {**_round_trip(model), field: value}

    with pytest.raises(ValueError, match=message):
        type(model).from_state_bundle(bundle)


def test_bundle_contains_no_training_rows(
    fitted: tuple[_Case, BaseAgeGroupModel], modeling_df: pd.DataFrame
) -> None:
    """The training frame is identified by hash; its rows are never copied.

    Fitted *summary* statistics are model state and legitimately appear (see
    ``age_group_prediction.state_bundle``). The check is therefore on what would
    reveal individual rows: no building ID, no array with one entry per
    training building, and no covariate value strictly inside its observed
    range, which no summary statistic reproduces.
    """
    _, model = fitted
    bundle = model.to_state_bundle()
    serialized = json.dumps(bundle)

    for building_id in modeling_df["building_id"]:
        # Whole numbers only: the digits also occur inside long float literals.
        standalone = rf"(?<![\d.]){int(building_id)}(?![\d.])"
        assert re.search(standalone, serialized) is None, building_id
    assert not _lists_of_length(bundle, len(modeling_df))
    for column in _FLOAT_COVARIATES:
        values = modeling_df[column].to_numpy(dtype=float)
        interior = values[(values > values.min()) & (values < values.max())]
        for value in interior:
            assert repr(float(value)) not in serialized, (column, value)
            assert f"{value:.17g}" not in serialized, (column, value)


def test_editing_an_exported_bundle_cannot_change_the_fitted_model(
    fitted: tuple[_Case, BaseAgeGroupModel],
) -> None:
    """Emptying an exported bundle in place leaves the next export unchanged."""
    _, model = fitted
    before = json.dumps(model.to_state_bundle(), sort_keys=True)

    _clear_every_container(model.to_state_bundle())

    assert json.dumps(model.to_state_bundle(), sort_keys=True) == before


def _lists_of_length(value: object, length: int) -> list[list[object]]:
    """Return every list, at any depth, with exactly ``length`` entries."""
    if isinstance(value, dict):
        return [found for item in value.values() for found in _lists_of_length(item, length)]
    if isinstance(value, list):
        own = [value] if len(value) == length else []
        return own + [found for item in value for found in _lists_of_length(item, length)]
    return []


def _clear_every_container(value: object) -> None:
    """Empty every nested dict and list in place, innermost first."""
    if isinstance(value, dict):
        for item in value.values():
            _clear_every_container(item)
        value.clear()
    elif isinstance(value, list):
        for item in value:
            _clear_every_container(item)
        value.clear()
