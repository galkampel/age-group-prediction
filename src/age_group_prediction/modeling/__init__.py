"""Age-group models as scikit-learn-style estimators, and the metrics that score them.

Each model subclasses :class:`BaseAgeGroupModel`: settings in the constructor,
``fit(X, y)`` and ``predict(X)``, and ``evaluate(y_true, y_pred, metric)`` with
a :class:`Metric`. This package replaces ``models/`` and ``modeling_config``,
which are deleted once every model is rebuilt here.
"""

from .base import BaseAgeGroupModel
from .direct_cohort import DirectCohortModel, Objective
from .metrics import MAE, POISSON_DEVIANCE, RMSE, Metric

__all__ = [
    "MAE",
    "POISSON_DEVIANCE",
    "RMSE",
    "BaseAgeGroupModel",
    "DirectCohortModel",
    "Metric",
    "Objective",
]
