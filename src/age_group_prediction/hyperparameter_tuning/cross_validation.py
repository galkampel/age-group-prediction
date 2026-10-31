"""The tuning of a composite model by its cross-validated score.

:class:`CrossValidationTuner` holds the study's design; each of its tune
methods runs one study of the model it names, whose objective draws the
components, builds the model and scores it on the folds. The total times
probability model has also a tune method per component, the other one fixed
(coordinate descent). Each trial fits its folds in ``n_jobs`` parallel
processes, and the trials run one after another (D13, D19).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from sklearn.model_selection import BaseCrossValidator

from ..feature_engineering import FeatureTransformer
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import Groups
from .aggregation import Aggregator, Mean
from .builders import (
    build_independent_cohorts_model,
    build_total_times_probability_model,
    suggest_count_component,
    suggest_independent_cohorts_components,
    suggest_probability_component,
    suggest_total_times_probability_components,
)
from .components import (
    CountComponent,
    CountFamily,
    ProbabilityComponent,
    ProbabilityFamily,
)

__all__ = ["CrossValidationTuner"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CrossValidationTuner:
    """One Optuna study per call on cross-validation folds; the study's design is fixed at construction.

    ``metric`` scores each fold and gives the study its direction;
    ``aggregator`` combines a trial's fold scores into its value. The first
    ``n_startup_trials`` of the ``n_trials`` are drawn at random, before TPE's
    model guides the draws. ``n_jobs`` processes fit a trial's folds.
    """

    cv: BaseCrossValidator
    metric: Metric
    n_trials: int
    n_startup_trials: int
    # The aggregator decides the selection (D9); the plain mean by default,
    # since the folds differ by one row at most.
    aggregator: Aggregator = dataclasses.field(default_factory=Mean)
    n_jobs: int = 1

    def fold_score(
        self,
        model: BaseAgeGroupModel,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        train_index: np.ndarray,
        val_index: np.ndarray,
    ) -> float:
        """The score on one fold's validation rows of ``model`` fitted on its training rows."""
        X_train, Y_train, exposure_train = (
            X.iloc[train_index],
            Y.iloc[train_index],
            exposure.iloc[train_index],
        )
        X_val, Y_val, exposure_val = (
            X.iloc[val_index],
            Y.iloc[val_index],
            exposure.iloc[val_index],
        )
        model.fit(X_train, Y_train, exposure=exposure_train)
        Y_val_pred = model.predict(X_val, exposure=exposure_val)
        return self.metric(Y_val, Y_val_pred)

    def cv_score(
        self,
        model: BaseAgeGroupModel,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        groups: Groups | None = None,
    ) -> float:
        """The aggregated fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state).

        The folds are fitted in parallel; the scores are the same as one
        after another, so a seeded study is unchanged. The processes do not
        inherit a script's ``-W`` filters: a warning in a fold raises there
        only under the environment's ``PYTHONWARNINGS`` (D15).
        """
        folds = list(self.cv.split(X, Y, groups))
        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fold_score)(model, X, Y, exposure, train_index, val_index)
            for train_index, val_index in folds
        )
        fold_sizes = [len(val_index) for _, val_index in folds]
        return self.aggregator.aggregate_scores(scores, fold_sizes)

    def tune_independent_cohorts_model(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        groups: Groups | None = None,
        *,
        family: CountFamily,
        use_exposure: bool,
        cohorts: Sequence[str],
        feature_transformer_dict: Mapping[str, FeatureTransformer],
        share_hyperparams: bool,
        share_feature_transformer: bool,
        y_min: float,
        random_state: int | None,
        study_seed: int,
    ) -> tuple[optuna.Study, float]:
        """One study of the independent cohorts model of one family and exposure, on the rows of ``X``; the study and its seconds.

        The settings are those of ``suggest_independent_cohorts_components``
        and ``build_independent_cohorts_model``.
        """

        def objective(trial: optuna.Trial) -> float:
            components = suggest_independent_cohorts_components(
                trial,
                family=family,
                use_exposure=use_exposure,
                cohorts=cohorts,
                feature_transformer_dict=feature_transformer_dict,
                share_hyperparams=share_hyperparams,
                share_feature_transformer=share_feature_transformer,
            )
            model = build_independent_cohorts_model(
                components, y_min=y_min, random_state=random_state
            )
            return self.cv_score(model, X, Y, exposure, groups)

        return self._optimize(objective, study_seed=study_seed)

    def tune_total_times_probability_model(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        groups: Groups | None = None,
        *,
        total_family: CountFamily,
        probability_family: ProbabilityFamily,
        use_exposure: bool,
        total_feature_transformer_dict: Mapping[str, FeatureTransformer],
        probability_feature_transformer_dict: Mapping[str, FeatureTransformer],
        y_min: float,
        random_state: int | None,
        study_seed: int,
    ) -> tuple[optuna.Study, float]:
        """One study of the total times probability model of one family pair and exposure, on the rows of ``X``; the study and its seconds.

        The settings are those of
        ``suggest_total_times_probability_components`` and
        ``build_total_times_probability_model``.
        """

        def objective(trial: optuna.Trial) -> float:
            total_component, probability_component = (
                suggest_total_times_probability_components(
                    trial,
                    total_family=total_family,
                    probability_family=probability_family,
                    use_exposure=use_exposure,
                    total_feature_transformer_dict=total_feature_transformer_dict,
                    probability_feature_transformer_dict=probability_feature_transformer_dict,
                )
            )
            model = build_total_times_probability_model(
                total_component,
                probability_component,
                y_min=y_min,
                random_state=random_state,
            )
            return self.cv_score(model, X, Y, exposure, groups)

        return self._optimize(objective, study_seed=study_seed)

    def tune_total_component(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        groups: Groups | None = None,
        *,
        total_family: CountFamily,
        use_exposure: bool,
        total_feature_transformer_dict: Mapping[str, FeatureTransformer],
        probability_component: ProbabilityComponent,
        y_min: float,
        random_state: int | None,
        study_seed: int,
    ) -> tuple[optuna.Study, float]:
        """One study of the total times probability model's total of one family and exposure, ``probability_component`` fixed; the study and its seconds.

        Coordinate descent's step for the total. Under the Poisson deviance
        the fixed probabilities add the same constant to every trial's
        value, so the best total is the joint study's; under another metric
        it is the best total given ``probability_component``.
        """

        def objective(trial: optuna.Trial) -> float:
            total_component = suggest_count_component(
                trial,
                name="total",
                family=total_family,
                use_exposure=use_exposure,
                feature_transformer_dict=total_feature_transformer_dict,
            )
            model = build_total_times_probability_model(
                total_component,
                probability_component,
                y_min=y_min,
                random_state=random_state,
            )
            return self.cv_score(model, X, Y, exposure, groups)

        return self._optimize(objective, study_seed=study_seed)

    def tune_probability_component(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        exposure: pd.Series,
        groups: Groups | None = None,
        *,
        total_component: CountComponent,
        probability_family: ProbabilityFamily,
        probability_feature_transformer_dict: Mapping[str, FeatureTransformer],
        y_min: float,
        random_state: int | None,
        study_seed: int,
    ) -> tuple[optuna.Study, float]:
        """One study of the total times probability model's probabilities of one family, ``total_component`` fixed; the study and its seconds.

        Coordinate descent's step for the probabilities. Under the Poisson
        deviance the fixed total adds the same constant to every trial's
        value, so the best probabilities are the joint study's; under
        another metric they are the best given ``total_component``.
        """

        def objective(trial: optuna.Trial) -> float:
            probability_component = suggest_probability_component(
                trial,
                name="probability",
                family=probability_family,
                feature_transformer_dict=probability_feature_transformer_dict,
            )
            model = build_total_times_probability_model(
                total_component,
                probability_component,
                y_min=y_min,
                random_state=random_state,
            )
            return self.cv_score(model, X, Y, exposure, groups)

        return self._optimize(objective, study_seed=study_seed)

    def _optimize(
        self, objective: Callable[[optuna.Trial], float], *, study_seed: int
    ) -> tuple[optuna.Study, float]:
        """The study of ``objective`` and its seconds.

        A trial whose fit raises RuntimeError (an NB2 fit that does not
        converge, re-raised as such from a fold's process) is marked FAIL and
        the study goes on; without ``catch`` one such trial would end the
        study.
        """
        study = optuna.create_study(
            direction=self.metric.direction,
            sampler=TPESampler(seed=study_seed, n_startup_trials=self.n_startup_trials),
        )
        start = time.perf_counter()
        study.optimize(objective, n_trials=self.n_trials, catch=(RuntimeError,))
        return study, time.perf_counter() - start
