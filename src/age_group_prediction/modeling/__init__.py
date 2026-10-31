"""Age-group models as scikit-learn-style estimators.

Each model subclasses :class:`BaseAgeGroupModel`: settings in the constructor,
``fit(X, y, exposure=None)`` and ``predict(X, exposure=None)``; a caller
scores the predictions with :mod:`~age_group_prediction.scoring`. This
package replaces
``models/`` and ``modeling_config``, which are deleted once every model is
rebuilt here.
"""

from .base import BaseAgeGroupModel
from .cohort_probability import (
    CalibrationMethod,
    Classifier,
    CohortProbabilityModel,
    ReplicationMethod,
)
from .count_model import CountModel, ExposureRegressor, Regressor
from .independent_cohorts import CohortModels, IndependentCohortModels
from .negative_binomial import NegativeBinomialRegressor
from .total_times_probability import TotalTimesProbabilityModel

__all__ = [
    "BaseAgeGroupModel",
    "CalibrationMethod",
    "Classifier",
    "CohortModels",
    "CohortProbabilityModel",
    "CountModel",
    "ExposureRegressor",
    "IndependentCohortModels",
    "NegativeBinomialRegressor",
    "Regressor",
    "ReplicationMethod",
    "TotalTimesProbabilityModel",
]
