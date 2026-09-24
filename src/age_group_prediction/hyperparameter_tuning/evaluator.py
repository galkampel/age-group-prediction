"""How one trial is scored: set the parameters once, then fit and score every fold.

The evaluator holds the settings; the data are arguments to
:meth:`CVHyperparameterEvaluator.evaluate`, so one evaluator can score many
datasets. Every call re-splits with ``cv``, a :meth:`Splitter.cv` validator,
which requires an int seed, so every trial is compared on identical folds.
Each trial gets its own copies of ``features`` (a ``FeatureTransformer``) and
the model, with the trial's parameters set; each fold refits them on its
training rows only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

import numpy as np
import optuna
from numpy.typing import ArrayLike
from optuna.trial import BaseTrial
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator

from ..feature_engineering import FeatureTransformer
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import DesignMatrix, Groups, Target, take_rows
from .parameters import Parameter

__all__ = ["Aggregation", "CVHyperparameterEvaluator"]

Aggregation = Literal["weighted_mean", "mean", "lower_bound"]


class CVHyperparameterEvaluator:
    """Score one hyperparameter set by cross-validation; greater is better.

    Optuna calls its objective with the trial alone, so bind the data::

        evaluator = CVHyperparameterEvaluator(
            DirectCohortModel(use_exposure=True), parameters,
            cv=splitter.cv(n_splits=5, random_state=42),
            metric=POISSON_DEVIANCE, features=tree,
        )
        study.optimize(lambda trial: evaluator.evaluate(
            trial, train_df, y_train, groups_train, exposure=n_train))

    A lower-is-better metric is negated, so the study always maximizes.
    ``features`` turns the raw table ``X`` into each fold's design matrix and
    is refitted on that fold's training rows, so validation rows can't leak
    into it. For the ``random`` method pass ``groups=None``: ``KFold`` warns
    on every split that receives groups. ``z`` applies only to
    ``"lower_bound"`` (default 1.0).
    """

    def __init__(
        self,
        model: BaseAgeGroupModel,
        parameters: Sequence[Parameter],
        *,
        cv: BaseCrossValidator,
        metric: Metric,
        features: FeatureTransformer,
        aggregation: Aggregation = "weighted_mean",
        z: float | None = None,
    ) -> None:
        if not isinstance(metric, Metric):
            # An old sklearn scoring string would otherwise fail only after the
            # first fold is fitted.
            raise TypeError(f"metric must be a Metric, got {metric!r}")
        if not isinstance(features, FeatureTransformer):
            # Any sklearn transformer would run; the repo's features are declared
            # through FeatureTransformer.
            raise TypeError(
                f"features must be a FeatureTransformer, got {type(features).__name__}"
            )
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

        self.model = model
        self.parameters = tuple(parameters)
        # Split in every call; Splitter.cv guarantees at least 2 non-empty
        # folds, identical on each call.
        self.cv = cv
        self.metric = metric
        self.features = features
        self.aggregation: Aggregation = aggregation
        self.z = 1.0 if aggregation == "lower_bound" and z is None else z

    def evaluate(
        self,
        trial: BaseTrial,
        X: DesignMatrix,
        y: Target,
        groups: Groups | None = None,
        *,
        exposure: ArrayLike | None = None,
    ) -> float:
        """Fit and score every fold with the trial's parameters.

        ``trial`` comes from a study, or is ``optuna.trial.FixedTrial(params)``
        to score one parameter set without a study. ``X`` is the raw table,
        a DataFrame for any ``features`` with plans (a numpy array works only
        with a pass-through transformer). ``exposure`` is each row's
        raw exposure, one per row of ``X``. It is sliced by position, like
        ``X``, and passed to the model's ``fit`` and ``predict``.
        """
        # Once, before any fold. Optuna would return the cached value on a
        # repeated suggest anyway, but every fold then visibly shares one set.
        params = {p.name: p.suggest(trial) for p in self.parameters}
        n = None if exposure is None else np.asarray(exposure)
        # cv.split checks X, y and groups agree; nothing else would catch an
        # exposure for other rows, since each fold slices it to the right size.
        if n is not None and n.shape != (len(X),):
            raise ValueError(
                f"exposure must hold one value per row of X ({len(X)}), "
                f"got shape {n.shape}"
            )
        features, model = self.build_features_and_model(params)
        total = weight = 0.0
        for fold, (train_index, validation_index) in enumerate(
            self.cv.split(X, y, groups)
        ):
            X_train, X_validation = (
                take_rows(X, train_index),
                take_rows(X, validation_index),
            )
            y_train = take_rows(y, train_index)
            # Refitted on this fold's training rows only, so no validation
            # statistics leak in; fit replaces the previous fold's state.
            features.fit(X_train, y_train)
            X_train, X_validation = (
                features.transform(X_train),
                features.transform(X_validation),
            )
            model.fit(X_train, y_train, exposure=None if n is None else n[train_index])
            predicted = model.predict(
                X_validation, exposure=None if n is None else n[validation_index]
            )
            value = model.evaluate(
                take_rows(y, validation_index), predicted, self.metric
            )
            # One direction everywhere: the study always maximizes.
            score = value if self.metric.greater_is_better else -value
            # Checked before report: Optuna stores a NaN intermediate silently.
            if not math.isfinite(score):
                raise ValueError(
                    f"trial {trial.number}, fold {fold}: score is {score}; "
                    "check the data and that the metric suits the model"
                )
            # The pruner sees the running mean: size-weighted, or plain for "mean".
            weight_k = 1 if self.aggregation == "mean" else len(validation_index)
            total += weight_k * score
            weight += weight_k
            trial.report(total / weight, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after fold {fold}")
        # TODO(2.3): "lower_bound" subtracts z * SE; record the user attrs.
        return total / weight

    def build_features_and_model(
        self, params: Mapping[str, Any]
    ) -> tuple[FeatureTransformer, BaseAgeGroupModel]:
        """Unfitted copies of the templates, with ``params`` set on the model.

        Called once per trial, and for the final refit. Copies, so the
        templates stay unfitted and parallel trials (Optuna ``n_jobs > 1``)
        share nothing. One pair serves every fold of a trial, because ``fit``
        replaces all fitted state.
        """
        model: BaseAgeGroupModel = clone(self.model).set_params(**params)
        return clone(self.features), model
