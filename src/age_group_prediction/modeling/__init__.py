"""Age-group models as scikit-learn-style estimators.

Each model subclasses :class:`BaseAgeGroupModel`: settings in the constructor,
``fit(X, y, exposure=None)`` and ``predict(X, exposure=None)``, and
``evaluate(y_true, y_pred, metric)`` with a
:class:`~age_group_prediction.scoring.Metric`. This package replaces
``models/`` and ``modeling_config``, which are deleted once every model is
rebuilt here.
"""

from .base import BaseAgeGroupModel
from .direct_cohort import DirectCohortModel, Objective
from .independent_cohorts import CohortModels, IndependentCohortModels
from .pipeline import ModelPipeline
from .total_children import Solver, TotalChildrenModel

__all__ = [
    "BaseAgeGroupModel",
    "CohortModels",
    "DirectCohortModel",
    "IndependentCohortModels",
    "ModelPipeline",
    "Objective",
    "Solver",
    "TotalChildrenModel",
]
