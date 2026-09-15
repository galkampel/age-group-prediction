"""Models-from-code pyfunc wrapper for one Gate 8 full-training refit.

Logged with ``mlflow.pyfunc.log_model(python_model=<this file's path>, ...)``:
MLflow executes this module at load time and reads the instance
``mlflow.models.set_model`` registers, rather than unpickling a stored
object (plan section 11, "no opaque model serialization"). The wrapped model
never touches MLflow itself; this file is the only place in the project where
a state bundle and MLflow meet.

The serving contract is deterministic point prediction only: building ID,
total mean, one ``{cohort}_mean`` column per cohort, and one
``{cohort}_probability`` column per cohort, in schema order. Predictive
bootstrap draws that need the training frame stay offline evaluation
evidence and are never exposed here.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow
import pandas as pd

from age_group_prediction.modeling_config import PredictionConfig
from age_group_prediction.models.base import BaseAgeGroupModel

# LightGBM before torch; see `age_group_prediction.models`'s import note.
# isort: off
from age_group_prediction.models.direct_cohort import DirectCohortModel
from age_group_prediction.models.bayesian_conditional import BayesianConditionalModel
# isort: on
from age_group_prediction.models.independent_total_probability import (
    IndependentTotalProbabilityModel,
)
from age_group_prediction.results import PredictionResult
from age_group_prediction.state_bundle import check_bundle_header

_MODEL_CLASSES: dict[str, type[BaseAgeGroupModel]] = {
    model_class.__name__: model_class
    for model_class in (
        DirectCohortModel,
        IndependentTotalProbabilityModel,
        BayesianConditionalModel,
    )
}


def prediction_to_frame(prediction: PredictionResult) -> pd.DataFrame:
    """Return the pyfunc output frame for one ``PredictionResult``.

    Shared by ``AgeGroupPyfuncModel.predict`` and ``tracking.final``'s reload
    equality check, so both sides build identical columns by construction
    rather than by keeping two implementations in sync by hand.
    """
    records: list[dict[str, Any]] = []
    for row_index, building_id in enumerate(prediction.building_ids):
        record: dict[str, Any] = {
            "building_id": building_id,
            "total_mean": float(prediction.total_mean[row_index]),
        }
        for cohort_index, cohort in enumerate(prediction.cohort_names):
            record[f"{cohort}_mean"] = float(
                prediction.cohort_means[row_index, cohort_index]
            )
            record[f"{cohort}_probability"] = float(
                prediction.age_group_probabilities[row_index, cohort_index]
            )
        records.append(record)
    return pd.DataFrame(records)


class AgeGroupPyfuncModel(mlflow.pyfunc.PythonModel):
    """Loadable point-prediction wrapper around one full-training state bundle."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Read the bundle artifact, validate its header, and rebuild the model.

        Dispatches on the bundle's own ``model_class`` field rather than a
        constructor argument, so the same script logs and loads all three
        approach winners. Never receives ``train_df``: point prediction needs
        no training rows.
        """
        with open(context.artifacts["state_bundle"], encoding="utf-8") as handle:
            bundle = json.load(handle)
        model_class_name = bundle.get("model_class")
        model_class = _MODEL_CLASSES.get(str(model_class_name))
        if model_class is None:
            raise ValueError(
                f"Unsupported state bundle model_class {model_class_name!r}; "
                f"expected one of {sorted(_MODEL_CLASSES)}"
            )
        check_bundle_header(
            bundle,
            model_class=model_class,
            implementation_version=model_class.implementation_version,
        )
        self._model: BaseAgeGroupModel = model_class.from_state_bundle(bundle)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return building ID, total mean, and per-cohort means/probabilities."""
        del context, params
        prediction = self._model.predict(model_input, prediction_config=PredictionConfig())
        return prediction_to_frame(prediction)


mlflow.models.set_model(AgeGroupPyfuncModel())
