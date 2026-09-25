"""How one trial is scored: set the parameters once, then fit and score every fold.

The evaluator holds the settings; the data are arguments to
:meth:`CVHyperparameterEvaluator.evaluate`, so one evaluator can score many
datasets. Every call re-splits with ``cv``, a :meth:`Splitter.cv` validator,
which requires an int seed, so every trial is compared on identical folds.
Each trial gets its own copies of ``feature_transformer`` and the model, with
the trial's parameters set; each fold refits them on its training rows only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import optuna
from optuna.trial import BaseTrial
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator

from ..feature_engineering import FeatureTransformer
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import DesignMatrix, Exposure, Groups, Target, take_rows
from .aggregation import Aggregation, WeightedMean
from .parameters import Parameter

__all__ = ["CVHyperparameterEvaluator"]


class CVHyperparameterEvaluator:
    """Score one hyperparameter set by cross-validation; greater is better.

    Optuna calls its objective with the trial alone, so bind the data::

        evaluator = CVHyperparameterEvaluator(
            DirectCohortModel(use_exposure=True), parameters,
            cv=splitter.cv(n_splits=5, random_state=42),
            metric=POISSON_DEVIANCE, feature_transformer=tree,
        )
        study.optimize(lambda trial: evaluator.evaluate(
            trial, train_df, y_train, groups_train, exposure=exposure_train))

    A lower-is-better metric is negated, so the study always maximizes.
    ``feature_transformer`` turns the raw table ``X`` into each fold's design
    matrix and is refitted on that fold's training rows, so validation rows
    can't leak into it. ``aggregation`` combines the fold scores into the
    trial value; the pruner sees each fold's own score. For the ``random``
    method pass ``groups=None``: ``KFold`` warns on every split that receives
    groups.
    """

    def __init__(
        self,
        model: BaseAgeGroupModel,
        parameters: Sequence[Parameter],
        *,
        cv: BaseCrossValidator,
        metric: Metric,
        feature_transformer: FeatureTransformer,
        aggregation: Aggregation = WeightedMean(),  # noqa: B008 (frozen: safe to share)
    ) -> None:
        if not isinstance(metric, Metric):
            # An old sklearn scoring string would otherwise fail only after the
            # first fold is fitted.
            raise TypeError(f"metric must be a Metric, got {metric!r}")
        if not isinstance(feature_transformer, FeatureTransformer):
            # Any sklearn transformer would run; the repo's features are declared
            # through FeatureTransformer.
            raise TypeError(
                "feature_transformer must be a FeatureTransformer, "
                f"got {type(feature_transformer).__name__}"
            )
        if not isinstance(aggregation, Aggregation):
            # A string ("mean") would otherwise fail only after every fold is fitted.
            raise TypeError(f"aggregation must be an Aggregation, got {aggregation!r}")

        if not parameters:
            raise ValueError("parameters must not be empty")
        # Optuna would silently hand the second parameter the first one's value.
        duplicates = sorted(
            name
            for name, count in Counter(p.name for p in parameters).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"parameter names must be unique, repeated: {duplicates}")

        self.model = model
        self.parameters = tuple(parameters)
        # Split in every call; Splitter.cv guarantees at least 2 non-empty
        # folds, identical on each call.
        self.cv = cv
        self.metric = metric
        self.feature_transformer = feature_transformer
        self.aggregation = aggregation

    def evaluate(
        self,
        trial: BaseTrial,
        X: DesignMatrix,
        y: Target,
        groups: Groups | None = None,
        *,
        exposure: Exposure | None = None,
    ) -> float:
        """Fit and score every fold with the trial's parameters.

        ``trial`` comes from a study, or is ``optuna.trial.FixedTrial(params)``
        to score one parameter set without a study. ``X`` is the raw table,
        a DataFrame for any ``feature_transformer`` with plans (a numpy array
        works only with a pass-through transformer). ``exposure``: one raw
        exposure per row of ``X``, sliced by position like ``X`` and passed to
        the model's ``fit`` and ``predict``.
        """
        self._check_exposure(exposure, X)
        params = {p.name: p.suggest(trial) for p in self.parameters}
        feature_transformer, model = self.build_feature_transformer_and_model(params)
        scores: list[float] = []
        fold_sizes: list[int] = []
        for fold, (train_index, val_index) in enumerate(self.cv.split(X, y, groups)):
            X_train, X_val = take_rows(X, train_index), take_rows(X, val_index)
            y_train, y_val = take_rows(y, train_index), take_rows(y, val_index)
            exposure_train, exposure_val = (
                (None, None)
                if exposure is None
                else (take_rows(exposure, train_index), take_rows(exposure, val_index))
            )
            # Refitted on this fold's training rows only, so no validation
            # statistics leak in; fit replaces the previous fold's state.
            feature_transformer.fit(X_train, y_train)
            X_train, X_val = (
                feature_transformer.transform(X_train),
                feature_transformer.transform(X_val),
            )
            model.fit(X_train, y_train, exposure=exposure_train)
            y_val_pred = model.predict(X_val, exposure=exposure_val)
            value = model.evaluate(y_val, y_val_pred, self.metric)
            # One direction everywhere: the study always maximizes.
            score = value if self.metric.greater_is_better else -value
            # Checked before report: Optuna stores a NaN intermediate silently.
            if not math.isfinite(score):
                raise ValueError(
                    f"trial {trial.number}, fold {fold}: score is {score}; "
                    "check the data and that the metric suits the model"
                )
            scores.append(score)
            fold_sizes.append(len(val_index))
            # The fold's own score, as Optuna's WilcoxonPruner expects: it pairs
            # fold k across trials.
            trial.report(score, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after fold {fold}")
        # A pruned trial's fold scores are already its intermediate values.
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_sizes", fold_sizes)
        return self.aggregation.aggregate(scores, fold_sizes)

    @staticmethod
    def _check_exposure(exposure: Exposure | None, X: DesignMatrix) -> None:
        """Reject an exposure that isn't one value per row of ``X``.

        ``cv.split`` checks only ``X``, ``y`` and ``groups``. Each fold would
        silently slice a longer exposure to size; a shorter one would fail
        mid-CV.
        """
        if exposure is not None and np.shape(exposure) != (len(X),):
            raise ValueError(
                f"exposure must hold one value per row of X ({len(X)}), "
                f"got shape {np.shape(exposure)}"
            )

    def build_feature_transformer_and_model(
        self, params: Mapping[str, Any]
    ) -> tuple[FeatureTransformer, BaseAgeGroupModel]:
        """Unfitted copies of the templates, with ``params`` set on the model.

        Called once per trial, and for the final refit. Copies, so the
        templates stay unfitted and parallel trials (Optuna ``n_jobs > 1``)
        share nothing. One pair serves every fold of a trial, because ``fit``
        replaces all fitted state.
        """
        model: BaseAgeGroupModel = clone(self.model).set_params(**params)
        return clone(self.feature_transformer), model
