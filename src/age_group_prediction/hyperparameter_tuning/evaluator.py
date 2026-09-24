"""How one trial is scored: set the parameters once, then fit and score every fold.

The evaluator holds the settings; the data are arguments to
:meth:`CVHyperparameterEvaluator.evaluate`, so one evaluator can score many
datasets. Every call re-splits with ``cv``, a :meth:`Splitter.cv` validator,
which requires an int seed, so every trial is compared on identical folds.
The model is a template: each fold fits ``clone(estimator).set_params(**params)``,
so a ``Pipeline``'s preprocessing is refitted on the fold's training rows only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, get_args

import optuna
from optuna.trial import BaseTrial
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator

from ..utils import DesignMatrix, Groups, Target, take_rows
from .parameters import Parameter, ParamValue

__all__ = ["Aggregation", "CVHyperparameterEvaluator"]

Aggregation = Literal["weighted_mean", "mean", "lower_bound"]


class CVHyperparameterEvaluator:
    """Score one hyperparameter set by cross-validation; greater is better.

    Optuna calls its objective with the trial alone, so bind the data::

        evaluator = CVHyperparameterEvaluator(
            pipeline, parameters, cv=splitter.cv(n_splits=5, random_state=42),
            scoring="neg_mean_poisson_deviance",
        )
        study.optimize(
            lambda trial: evaluator.evaluate(trial, X_train, y_train, groups_train)
        )

    For the ``random`` method pass ``groups=None``: ``KFold`` warns on every
    split that receives groups. ``scoring`` is required: sklearn would
    otherwise fall back silently to ``estimator.score``. ``z`` applies only to
    ``"lower_bound"`` (default 1.0).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        parameters: Sequence[Parameter],
        *,
        cv: BaseCrossValidator,
        scoring: str | Callable[..., float],
        aggregation: Aggregation = "weighted_mean",
        z: float | None = None,
    ) -> None:
        if aggregation not in get_args(Aggregation):
            raise ValueError(
                f"unknown aggregation {aggregation!r}; "
                f"expected {list(get_args(Aggregation))}"
            )
        if z is not None and aggregation != "lower_bound":
            raise ValueError(f"z applies only to 'lower_bound', not {aggregation!r}")
        if z is not None and not (math.isfinite(z) and z > 0):
            raise ValueError(f"z must be a positive finite number, got {z}")

        if not parameters:
            raise ValueError("parameters must not be empty")
        # Optuna would silently hand the second parameter the first one's value.
        duplicates = sorted(
            n for n, c in Counter(p.name for p in parameters).items() if c > 1
        )
        if duplicates:
            raise ValueError(f"parameter names must be unique, repeated: {duplicates}")

        self.estimator = estimator
        self.parameters = tuple(parameters)
        self.aggregation: Aggregation = aggregation
        self.z = 1.0 if aggregation == "lower_bound" and z is None else z
        # Split in every call; Splitter.cv guarantees at least 2 non-empty
        # folds, identical on each call.
        self.cv = cv
        self._scorer = check_scoring(estimator, scoring=scoring)

    def evaluate(
        self,
        trial: BaseTrial,
        X: DesignMatrix,
        y: Target,
        groups: Groups | None = None,
    ) -> float:
        """Fit and score every fold with the trial's parameters.

        ``trial`` comes from a study, or is ``optuna.trial.FixedTrial(params)``
        to score one parameter set without a study.
        """
        # Once, before any fold. Optuna would return the cached value on a
        # repeated suggest anyway, but every fold then visibly shares one set.
        params = {p.name: p.suggest(trial) for p in self.parameters}
        total = weight = 0.0
        for fold, (train_index, validation_index) in enumerate(
            self.cv.split(X, y, groups)
        ):
            model = self.build_estimator(params)
            model.fit(take_rows(X, train_index), take_rows(y, train_index))
            score = float(
                self._scorer(
                    model,
                    take_rows(X, validation_index),
                    take_rows(y, validation_index),
                )
            )
            # Checked before report: Optuna stores a NaN intermediate silently.
            if not math.isfinite(score):
                raise ValueError(
                    f"trial {trial.number}, fold {fold}: score is {score}; "
                    "check the data and that the scorer suits the model"
                )
            # The pruner sees the running mean: size-weighted, or plain for "mean".
            n = 1 if self.aggregation == "mean" else len(validation_index)
            total += n * score
            weight += n
            trial.report(total / weight, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after fold {fold}")
        # TODO(2.3): "lower_bound" subtracts z * SE; record the user attrs.
        return total / weight

    def build_estimator(self, params: Mapping[str, ParamValue]) -> BaseEstimator:
        """An unfitted copy of the template with ``params`` set; the template is untouched."""
        estimator: BaseEstimator = clone(self.estimator).set_params(**params)
        return estimator
