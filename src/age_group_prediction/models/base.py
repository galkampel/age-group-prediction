"""Shared lifecycle for age-group models."""

from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any, Self

import numpy as np
import pandas as pd

from ..evaluation import evaluate_predictions
from ..fitted_features import FittedFeatureTransformer
from ..hashing import column_schema_hash, table_hash
from ..metrics import Metric
from ..modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_SEED,
    FeatureSpec,
    ModelingSchema,
    PredictionConfig,
    PredictionValidationConfig,
)
from ..results import (
    EvaluationResult,
    ParametricDistributionSpec,
    PredictionResult,
)
from ..state_bundle import (
    STATE_BUNDLE_FORMAT,
    check_bundle_header,
    dependency_versions,
    restore_config,
    verify_training_frame,
)

__all__ = [
    "BaseAgeGroupModel",
    "EvaluationResult",
    "ParametricDistributionSpec",
    "PredictionResult",
]


class BaseAgeGroupModel(ABC):
    """Own fitted preprocessing and enforce the common model lifecycle."""

    contract_version = "1"
    # Each model bumps its own version when the meaning of its fitted state
    # changes. A state bundle written under a different version is refused.
    implementation_version = "unspecified"
    # Distributions, beyond the shared preprocessing stack, whose versions a
    # state bundle records for provenance.
    _bundle_dependencies: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
        default_rng_seed: int = DEFAULT_SEED,
        default_prediction_config: PredictionConfig | None = None,
        prediction_validation_config: PredictionValidationConfig | None = None,
    ) -> None:
        """Validate the seed, store shared configuration, and start unfitted."""
        if default_rng_seed < 0:
            raise ValueError("default_rng_seed must be nonnegative")
        self.schema = schema
        self.default_rng_seed = default_rng_seed
        self.default_prediction_config = (
            default_prediction_config or PredictionConfig()
        )
        self.prediction_validation_config = (
            prediction_validation_config or PredictionValidationConfig()
        )
        self._feature_spec: FeatureSpec | None = None
        self._feature_transformer: FittedFeatureTransformer | None = None
        self._operation_durations: dict[str, float] = {}
        self._training_data_hash: str | None = None
        self._training_schema_hash: str | None = None
        self._derived_seeds: list[dict[str, object]] = []
        self._active_operation: str | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether a fit or bundle load has completed (fitted preprocessing is set)."""
        return self._feature_transformer is not None

    @property
    def feature_transformer(self) -> FittedFeatureTransformer:
        """The fitted preprocessing transformer; raises if the model is unfitted."""
        self._require_fitted("access fitted preprocessing")
        assert self._feature_transformer is not None
        return self._feature_transformer

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        feature_spec: FeatureSpec,
        rng: np.random.Generator | None = None,
    ) -> BaseAgeGroupModel:
        """Fit preprocessing and model-specific state on one training partition."""
        self._clear_fitted_state()
        feature_spec.validate_for_schema(self.schema)
        resolved_rng = self._resolve_rng(rng, default_rng_seed=self.default_rng_seed)
        try:
            with self._operation_scope("fit"):
                # Recorded so a fitted model can be tied back to the exact rows
                # and column schema it learned from.
                self._training_data_hash = table_hash(
                    train_df, id_column=self.schema.building_id_column
                )
                self._training_schema_hash = column_schema_hash(train_df)
                selected_feature_spec = self._select_feature_spec(
                    train_df=train_df,
                    feature_spec=feature_spec,
                    rng=resolved_rng,
                )
                selected_feature_spec.validate_for_schema(self.schema)
                transformer = FittedFeatureTransformer(
                    selected_feature_spec, schema=self.schema
                )
                features = transformer.fit_transform(train_df)
                # Exposed before `_fit_model` so subclasses needing their own
                # fold-fitted transformers (e.g. internal CV tuning) can reuse
                # the same immutable spec. `is_fitted` stays False until the
                # final transformer is assigned below.
                self._feature_spec = selected_feature_spec
                self._fit_model(
                    train_df=train_df,
                    features=features,
                    log_exposure=transformer.get_log_exposure(train_df),
                    rng=resolved_rng,
                )
        except Exception:
            self._clear_fitted_state()
            raise
        self._feature_transformer = transformer
        return self

    def _select_feature_spec(
        self,
        *,
        train_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        rng: np.random.Generator,
    ) -> FeatureSpec:
        """Select a final feature specification before full-data preprocessing."""
        del train_df, rng
        return feature_spec

    def predict(
        self,
        eval_df: pd.DataFrame,
        *,
        prediction_config: PredictionConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> PredictionResult:
        """Predict with model and preprocessing state learned during fit."""
        self._require_fitted("predict")
        transformer = self.feature_transformer
        resolved_rng = self._resolve_rng(rng, default_rng_seed=self.default_rng_seed)
        with self._operation_scope("predict"):
            result = self._predict_model(
                eval_df=eval_df,
                features=transformer.transform(eval_df),
                log_exposure=transformer.get_log_exposure(eval_df),
                prediction_config=prediction_config or self.default_prediction_config,
                rng=resolved_rng,
            )
        if not isinstance(result, PredictionResult):
            raise TypeError("Model prediction hook must return PredictionResult")
        if result.validation_config != self.prediction_validation_config:
            # `replace` re-runs __post_init__, so invariants are re-checked
            # under this model's policy, and no field can be dropped if
            # PredictionResult gains one.
            result = replace(
                result, validation_config=self.prediction_validation_config
            )
        return result

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        *,
        metrics: Sequence[Metric],
        rng: np.random.Generator | None = None,
    ) -> EvaluationResult:
        """Evaluate through the shared result boundary without refitting.

        This calls `predict` internally, so `evaluate_duration_seconds`
        contains `predict_duration_seconds` rather than partitioning with it.
        """
        self._require_fitted("evaluate")
        resolved_rng = self._resolve_rng(rng, default_rng_seed=self.default_rng_seed)
        with self._operation_scope("evaluate"):
            prediction = self.predict(eval_df, rng=resolved_rng)
            result = evaluate_predictions(eval_df, prediction, metrics, rng=resolved_rng)
        return result

    def get_metadata(self) -> dict[str, object]:
        """Return JSON-serializable model and fitted-preprocessing metadata."""
        model_metadata = copy.deepcopy(dict(self._get_model_metadata()))
        metadata: dict[str, object] = {
            "contract_version": self.contract_version,
            "model_class": type(self).__name__,
            "is_fitted": self.is_fitted,
            "model": model_metadata,
            "feature_spec": (
                asdict(self._feature_spec) if self._feature_spec is not None else None
            ),
            "preprocessing": (
                self._feature_transformer.get_metadata()
                if self._feature_transformer is not None
                else None
            ),
            "training_data_hash": self._training_data_hash,
            "training_schema_hash": self._training_schema_hash,
            "fit_duration_seconds": self._operation_durations.get("fit"),
            "predict_duration_seconds": self._operation_durations.get("predict"),
            "evaluate_duration_seconds": self._operation_durations.get("evaluate"),
            "derived_seeds": copy.deepcopy(self._derived_seeds),
        }
        json.dumps(metadata)
        return metadata

    def configuration_record(self) -> dict[str, object]:
        """Return the JSON-safe configuration this model was constructed with.

        This covers everything that decides how the model fits and predicts,
        and nothing it learned: the shared constructor arguments plus the
        subclass's own ``_model_configuration``. Two models built by the same
        factory give equal records, fitted or not, so Gate 8 can fingerprint
        what a final attempt refits without depending on fitted values.
        """
        record: dict[str, object] = {
            "model_class": type(self).__name__,
            "implementation_version": self.implementation_version,
            "schema": asdict(self.schema),
            "default_rng_seed": self.default_rng_seed,
            "default_prediction_config": asdict(self.default_prediction_config),
            "prediction_validation_config": asdict(self.prediction_validation_config),
            "model": copy.deepcopy(dict(self._model_configuration())),
        }
        json.dumps(record, allow_nan=False)
        return record

    def to_state_bundle(self) -> dict[str, object]:
        """Return a JSON-safe bundle that rebuilds this fitted model.

        Plan section 11 requires a reconstructable artifact with a tested loader
        rather than opaque pickling. The bundle holds configuration, fitted
        preprocessing, and each model's fitted parameters as plain JSON values,
        and **no training rows**: the training frame is identified only by its
        recorded hashes. The layout is documented in
        ``age_group_prediction.state_bundle``; the model-specific part comes
        from ``_export_model_state``.
        """
        self._require_fitted("export a state bundle")
        bundle: dict[str, object] = {
            "bundle_format": STATE_BUNDLE_FORMAT,
            "model_class": type(self).__name__,
            "implementation_version": self.implementation_version,
            "schema": asdict(self.schema),
            "default_rng_seed": self.default_rng_seed,
            "feature_transformer": self.feature_transformer.to_state(),
            "training_data_hash": self._training_data_hash,
            "training_schema_hash": self._training_schema_hash,
            "dependency_versions": dependency_versions(
                "numpy", "pandas", "scikit-learn", *self._bundle_dependencies
            ),
            # Deep-copied so a caller editing the bundle cannot reach into the
            # fitted model's own dictionaries.
            "model_state": copy.deepcopy(self._export_model_state()),
        }
        # Fail here rather than at write time if a field ever stops being
        # JSON-safe. `allow_nan=False` because NaN and Infinity are not valid
        # JSON, and a strict consumer of the artifact would reject them.
        json.dumps(bundle, allow_nan=False)
        return bundle

    @classmethod
    def from_state_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        train_df: pd.DataFrame | None = None,
        default_prediction_config: PredictionConfig | None = None,
        prediction_validation_config: PredictionValidationConfig | None = None,
    ) -> Self:
        """Rebuild a fitted model from ``to_state_bundle`` output, without refitting.

        The bundle alone supports point predictions, pointwise log
        probabilities, and parametric distributions. ``train_df`` is needed
        only by models whose predictive draws or intervals refit on the
        training frame (the neighborhood-cluster bootstrap models); when given,
        it must hash to the values the bundle records, so uncertainty is never
        silently computed from a different partition.
        """
        check_bundle_header(
            bundle, model_class=cls, implementation_version=cls.implementation_version
        )
        schema = restore_config(DEFAULT_MODELING_SCHEMA, bundle["schema"])
        transformer = FittedFeatureTransformer.from_state(
            bundle["feature_transformer"], schema=schema
        )
        if train_df is not None:
            verify_training_frame(train_df, bundle=bundle, schema=schema)

        model = cls._from_model_state(
            bundle["model_state"],
            schema=schema,
            default_rng_seed=int(bundle["default_rng_seed"]),
            default_prediction_config=default_prediction_config,
            prediction_validation_config=prediction_validation_config,
        )
        model._feature_spec = transformer.feature_spec
        model._training_data_hash = str(bundle["training_data_hash"])
        model._training_schema_hash = str(bundle["training_schema_hash"])
        if train_df is not None:
            model._attach_training_frame(train_df)
        # Assigned last: `is_fitted` keys off this attribute, so the model must
        # not appear fitted until every other piece of state is in place.
        model._feature_transformer = transformer
        return model

    def _derive_backend_seed(
        self,
        rng: np.random.Generator,
        *,
        purpose: str,
    ) -> int:
        """Derive and record an integer seed for a library-specific RNG."""
        if not purpose:
            raise ValueError("Backend-seed purpose must not be empty")
        max_uint32 = int(np.iinfo(np.uint32).max)
        seed = int(rng.integers(0, max_uint32 + 1, dtype=np.uint32))
        self._derived_seeds.append(
            {
                "operation": self._active_operation or "model",
                "purpose": purpose,
                "seed": seed,
            }
        )
        return seed

    @contextmanager
    def _operation_scope(self, operation: str) -> Iterator[None]:
        """Scope one lifecycle operation: active name, seeds, and duration.

        Seeds recorded by a previous run of the *same* operation are dropped on
        entry, so repeating `predict` replaces its seed records instead of
        appending duplicates. Seeds from other operations are untouched, so a
        later `predict` never erases what `fit` recorded. The duration is
        committed after the `try/finally`, so a failed operation records none.
        """
        previous_operation = self._active_operation
        self._active_operation = operation
        self._derived_seeds = [
            record
            for record in self._derived_seeds
            if record["operation"] != operation
        ]
        started = time.perf_counter()
        try:
            yield
        finally:
            self._active_operation = previous_operation
        self._operation_durations[operation] = time.perf_counter() - started

    @staticmethod
    def _resolve_rng(
        rng: np.random.Generator | None,
        *,
        default_rng_seed: int,
    ) -> np.random.Generator:
        """Return ``rng``, or a fresh generator seeded with ``default_rng_seed``."""
        return np.random.default_rng(default_rng_seed) if rng is None else rng

    def _require_fitted(self, operation: str) -> None:
        """Raise ``RuntimeError`` naming ``operation`` if the model is unfitted."""
        if not self.is_fitted:
            raise RuntimeError(f"Model must be fitted before {operation}")

    def _clear_fitted_state(self) -> None:
        """Drop base-owned fitted state and records, then reset subclass state."""
        self._feature_spec = None
        self._feature_transformer = None
        self._training_data_hash = None
        self._training_schema_hash = None
        self._operation_durations = {}
        self._derived_seeds = []
        self._active_operation = None
        self._reset_model_state()

    @abstractmethod
    def _reset_model_state(self) -> None:
        """Clear subclass-owned fitted state before a fit attempt."""

    def _model_configuration(self) -> Mapping[str, object]:
        """Return subclass-owned constructor configuration for ``configuration_record``.

        Values must be JSON-safe and must not include anything learned by
        fitting. Base-owned configuration (schema, seed, prediction settings)
        is added by ``configuration_record`` and must not be repeated here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare its model configuration"
        )

    def _export_model_state(self) -> dict[str, object]:
        """Return subclass-owned fitted state and configuration for a bundle.

        Values must be JSON-safe. Base-owned state (schema, seed, fitted
        preprocessing, training hashes) is written by ``to_state_bundle`` and
        must not be repeated here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support state bundles"
        )

    @classmethod
    def _from_model_state(cls, state: Mapping[str, Any], **base_arguments: Any) -> Self:
        """Construct an instance and restore ``_export_model_state`` output.

        ``base_arguments`` are the base constructor's keyword arguments; pass
        them through to ``cls(...)`` together with the model's own restored
        configuration. Base-owned fitted state is restored by the caller.
        """
        raise NotImplementedError(f"{cls.__name__} does not support state bundles")

    def _attach_training_frame(self, train_df: pd.DataFrame) -> None:
        """Keep a hash-verified training frame after loading from a bundle.

        Models whose predictive draws refit on the training frame override this.
        """
        del train_df

    @staticmethod
    def _require_training_frame(train_df: pd.DataFrame | None) -> pd.DataFrame:
        """Return the retained training frame, or explain why it is missing.

        A fitted model always holds one. A model loaded from a state bundle
        holds one only if the caller passed ``train_df``.
        """
        if train_df is None:
            raise RuntimeError(
                "Predictive draws and intervals refit this model on its training "
                "frame, which a state bundle does not contain. Reload with "
                "from_state_bundle(bundle, train_df=<original training frame>) "
                "to request them; point predictions need no frame."
            )
        return train_df

    def _get_model_metadata(self) -> Mapping[str, object]:
        """Return placeholder model metadata with the keys subclasses fill in."""
        return {
            "implementation_version": "unspecified",
            "likelihood": "unspecified",
            "parameterization": "unspecified",
            "hyperparameters": {},
            "priors": None,
            "calibration": None,
            "dependency_versions": {},
            "uncertainty_method": "unspecified",
            "diagnostics": {},
        }

    @abstractmethod
    def _fit_model(
        self,
        *,
        train_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        rng: np.random.Generator,
    ) -> None:
        """Fit model-specific state from base-owned preprocessing output.

        `self._feature_spec` is already set when this runs, so subclasses
        that need their own fold-fitted transformers (e.g. internal CV
        tuning) can build them from the same immutable spec.
        """

    @abstractmethod
    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        """Return model-specific predictions through the shared result type."""