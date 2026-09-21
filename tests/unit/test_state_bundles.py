"""The building blocks every state bundle is made of.

Model-level round trips live in ``test_model_state_bundles.py``. These tests pin
the shared pieces: fitted preprocessing state, configuration restoration, the
bundle header and training-frame checks, and tuning evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    CategoricalFeatureSpec,
    FittedFeatureTransformer,
    build_modeling_table,
)
from age_group_prediction.experiment.candidates import enumerate_feature_specs
from age_group_prediction.hashing import column_schema_hash, table_hash
from age_group_prediction.modeling_config import (
    DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
    DEFAULT_DIRECT_COHORT_CONFIG,
    DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG,
)
from age_group_prediction.state_bundle import (
    STATE_BUNDLE_FORMAT,
    check_bundle_header,
    restore_config,
    restore_feature_spec,
    verify_training_frame,
)
from age_group_prediction.tuning import TrialRecord, TuningResult
from student_simulator import StudentPopulationSimulator, load_simulation_config


@pytest.fixture(scope="module")
def modeling_df() -> pd.DataFrame:
    """Return the modeling table built from the stage-1 simulation config."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    config = load_simulation_config(config_path)
    return build_modeling_table(StudentPopulationSimulator(config).run())


# Every predeclared candidate spec, including the opt-in daycare curve, plus the
# one unknown-category policy the candidates never vary.
_DECLARED_SPECS = [
    (candidate.name, candidate.spec)
    for component in ("tree", "total_count", "composition")
    for candidate in enumerate_feature_specs(component, include_daycare_saturation=True)
] + [
    (
        "composition__treat_unknown_as_reference",
        replace(DEFAULT_PROBABILITY_FEATURE_SPEC, unknown_category_policy="treat_as_reference"),
    )
]


@pytest.mark.parametrize(
    "spec",
    [spec for _, spec in _DECLARED_SPECS],
    ids=[name for name, _ in _DECLARED_SPECS],
)
def test_transformer_state_rebuilds_identical_preprocessing(
    modeling_df: pd.DataFrame, spec: object
) -> None:
    """Every declared spec's JSON-restored transformer matches the original exactly."""
    fit_rows, later_rows = modeling_df.iloc[:150], modeling_df.iloc[150:]
    original = FittedFeatureTransformer(spec).fit(fit_rows)  # type: ignore[arg-type]

    rebuilt = FittedFeatureTransformer.from_state(json.loads(json.dumps(original.to_state())))

    pd.testing.assert_frame_equal(
        rebuilt.transform(later_rows), original.transform(later_rows), check_exact=True
    )
    assert rebuilt.get_feature_names_out() == original.get_feature_names_out()
    assert json.loads(json.dumps(rebuilt.get_metadata())) == json.loads(
        json.dumps(original.get_metadata())
    )


def test_transformer_state_that_contradicts_its_spec_is_refused(
    modeling_df: pd.DataFrame,
) -> None:
    """State missing the scaling means its spec requires is refused on load."""
    state = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(modeling_df).to_state()
    state["numeric_means"] = None

    with pytest.raises(ValueError, match="scaling"):
        FittedFeatureTransformer.from_state(state)


def test_transformer_state_does_not_depend_on_json_key_order(
    modeling_df: pd.DataFrame,
) -> None:
    """A writer that sorts keys must not reorder the transformed columns."""
    original = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(modeling_df)
    state = json.loads(json.dumps(original.to_state(), sort_keys=True))

    rebuilt = FittedFeatureTransformer.from_state(state)

    pd.testing.assert_frame_equal(
        rebuilt.transform(modeling_df), original.transform(modeling_df), check_exact=True
    )

    del state["numeric_means"][DEFAULT_TOTAL_FEATURE_SPEC.numeric_features[0]]
    with pytest.raises(ValueError, match="numeric_means"):
        FittedFeatureTransformer.from_state(state)


def test_categorical_levels_follow_the_spec_column_order_not_the_schema_order(
    modeling_df: pd.DataFrame,
) -> None:
    """Each encoded column must use its own levels, whatever order declares them.

    The schema lists ``zone`` before ``school_status``; the spec lists them the
    other way round. Encoding by schema order used to pair each column with the
    other column's levels.
    """
    schema = replace(
        DEFAULT_MODELING_SCHEMA,
        categorical_feature_specs=(
            CategoricalFeatureSpec(
                column="zone", categories=("rural", "urban"), reference_category="rural"
            ),
            *DEFAULT_MODELING_SCHEMA.categorical_feature_specs,
        ),
    )
    spec = replace(
        DEFAULT_PROBABILITY_FEATURE_SPEC, categorical_features=("school_status", "zone")
    )
    frame = modeling_df.assign(
        zone=np.where(np.arange(len(modeling_df)) % 3 == 0, "urban", "rural")
    )

    original = FittedFeatureTransformer(spec, schema=schema).fit(frame)
    transformed = original.transform(frame)

    np.testing.assert_array_equal(
        transformed["zone_urban"].to_numpy(), (frame["zone"] == "urban").to_numpy(float)
    )
    np.testing.assert_array_equal(
        transformed["school_status_existing"].to_numpy(),
        (frame["school_status"] == "existing").to_numpy(float),
    )
    rebuilt = FittedFeatureTransformer.from_state(original.to_state(), schema=schema)
    pd.testing.assert_frame_equal(rebuilt.transform(frame), transformed, check_exact=True)


@pytest.mark.parametrize(
    "config",
    [
        DEFAULT_MODELING_SCHEMA,
        DEFAULT_TREE_FEATURE_SPEC,
        DEFAULT_DIRECT_COHORT_CONFIG,
        DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG,
        DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
    ],
    ids=lambda config: type(config).__name__,
)
def test_configs_come_back_from_json_as_equal_validated_instances(config: object) -> None:
    """Each default config survives a JSON round trip as an equal instance."""
    payload = json.loads(json.dumps(asdict(config)))  # type: ignore[call-overload]

    assert restore_config(config, payload) == config


def test_configs_restore_from_asdict_output_without_a_json_step() -> None:
    """In-memory bundles keep tuples of dataclasses as tuples of dicts."""
    restored = restore_config(DEFAULT_MODELING_SCHEMA, asdict(DEFAULT_MODELING_SCHEMA))

    assert restored == DEFAULT_MODELING_SCHEMA


@pytest.mark.parametrize("edit", ["missing field", "unknown field"])
def test_a_payload_must_name_exactly_the_config_fields(edit: str) -> None:
    """A default silently filling a missing field would rebuild another config."""
    payload = asdict(DEFAULT_TOTAL_FEATURE_SPEC)
    if edit == "missing field":
        del payload["scale_numeric"]
    else:
        payload["unexpected"] = 1

    with pytest.raises(ValueError, match="Cannot rebuild FeatureSpec"):
        restore_feature_spec(payload)


def test_any_feature_spec_restores_from_one_template() -> None:
    """``restore_feature_spec`` rebuilds tree and probability specs from JSON."""
    for spec in (DEFAULT_TREE_FEATURE_SPEC, DEFAULT_PROBABILITY_FEATURE_SPEC):
        assert restore_feature_spec(json.loads(json.dumps(asdict(spec)))) == spec


def test_a_tampered_config_is_rejected_by_its_own_validation() -> None:
    """An invalid field value is rejected by the spec's own validation on restore."""
    payload = json.loads(json.dumps(asdict(DEFAULT_TOTAL_FEATURE_SPEC)))
    payload["ses_form"] = "cubic"

    with pytest.raises(ValueError, match="SES form"):
        restore_feature_spec(payload)


class _Example:
    """Stands in for a model class; the header check reads only its name."""


def _header(**overrides: str) -> dict[str, str]:
    """Return a bundle header matching ``_Example``, with ``overrides`` applied."""
    return {
        "bundle_format": STATE_BUNDLE_FORMAT,
        "model_class": "_Example",
        "implementation_version": "2",
        **overrides,
    }


def test_a_matching_bundle_header_is_accepted() -> None:
    """A header matching format, class name, and version passes the check."""
    check_bundle_header(_header(), model_class=_Example, implementation_version="2")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bundle_format": "1"}, "bundle format"),
        ({"model_class": "OtherModel"}, "written by"),
        ({"implementation_version": "1"}, "implementation version"),
    ],
)
def test_a_mismatched_bundle_header_is_refused_by_name(
    overrides: dict[str, str], message: str
) -> None:
    """Each mismatched header field is refused with a message naming it."""
    with pytest.raises(ValueError, match=message):
        check_bundle_header(
            _header(**overrides), model_class=_Example, implementation_version="2"
        )


def test_training_frame_check_ignores_row_order_but_not_content(
    modeling_df: pd.DataFrame,
) -> None:
    """A shuffled training frame verifies; one with a changed value is refused."""
    recorded = {
        "training_data_hash": table_hash(modeling_df, id_column="building_id"),
        "training_schema_hash": column_schema_hash(modeling_df),
    }
    shuffled = modeling_df.sample(frac=1.0, random_state=0)
    verify_training_frame(shuffled, bundle=recorded, schema=DEFAULT_MODELING_SCHEMA)

    changed = modeling_df.copy()
    changed.loc[changed.index[0], "ses"] += 1.0
    with pytest.raises(ValueError, match="training_data_hash"):
        verify_training_frame(changed, bundle=recorded, schema=DEFAULT_MODELING_SCHEMA)


def test_tuning_result_comes_back_from_json_with_its_tuple_structure() -> None:
    """A ``TuningResult`` round-trips through JSON with its tuples restored."""
    result = TuningResult(
        best_params={"learning_rate": 0.1, "max_depth": 3},
        best_value=1.25,
        best_trial_number=1,
        trials=(
            TrialRecord(0, "PRUNED", 1.5, {"learning_rate": 0.2, "max_depth": 2}, ((0, 1.5),)),
            TrialRecord(
                1,
                "COMPLETE",
                1.25,
                {"learning_rate": 0.1, "max_depth": 3},
                ((0, 1.3), (1, 1.25)),
            ),
        ),
    )

    assert TuningResult.from_dict(json.loads(json.dumps(asdict(result)))) == result
