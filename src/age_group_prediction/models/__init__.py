"""Shared contracts and concrete age-group model implementations."""

from ..results import EvaluationResult, ParametricDistributionSpec, PredictionResult
from .base import BaseAgeGroupModel

# `direct_cohort` (LightGBM) must be imported before `bayesian_conditional`
# (torch). Each ships its own OpenMP runtime, and with torch's loaded first,
# LightGBM segfaults restoring a booster in a fresh process -- exactly what
# loading a logged models-from-code pyfunc does. An in-process reload cannot
# catch this, because LightGBM is already initialized there.
# isort: off
from .direct_cohort import DirectCohortModel
from .bayesian_conditional import BayesianConditionalModel
# isort: on
from .independent_total_probability import IndependentTotalProbabilityModel

__all__ = [
	"BaseAgeGroupModel",
	"BayesianConditionalModel",
	"DirectCohortModel",
	"EvaluationResult",
	"IndependentTotalProbabilityModel",
	"ParametricDistributionSpec",
	"PredictionResult",
]