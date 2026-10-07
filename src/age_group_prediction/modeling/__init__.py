"""Age-group models as scikit-learn-style estimators.

Each model subclasses :class:`BaseAgeGroupModel`: settings in the constructor,
``fit(X, y, exposure=None)`` and ``predict(X, exposure=None)``, and
``evaluate(y_true, y_pred, metric)`` with a
:class:`~age_group_prediction.scoring.Metric`. This package replaces
``models/`` and ``modeling_config``, which are deleted once every model is
rebuilt here.
"""

from .base import BaseAgeGroupModel
from .cohort_probability import Classifier, CohortProbabilityModel, ReplicationType
from .count_model import CountModel, ExposureRegressor, Regressor
from .independent_cohorts import CohortModels, IndependentCohortModels
from .negative_binomial import NegativeBinomialRegressor
from .optimization import Solver
from .total_children import TotalChildrenModel

__all__ = [
    "BaseAgeGroupModel",
    "Classifier",
    "CohortModels",
    "CohortProbabilityModel",
    "CountModel",
    "ExposureRegressor",
    "IndependentCohortModels",
    "NegativeBinomialRegressor",
    "Regressor",
    "ReplicationType",
    "Solver",
    "TotalChildrenModel",
]
