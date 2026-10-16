"""Tests for ``PermutationImportance``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from age_group_prediction.feature_engineering import (
    ColumnPlan,
    FeatureTransformer,
    OneHot,
    Standardize,
)
from age_group_prediction.inspection import PermutationImportance
from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    CountModel,
    TotalTimesProbabilityModel,
)
from age_group_prediction.preprocessing import ExposureTransformer
from age_group_prediction.scoring import COHORT_POISSON_DEVIANCE, Metric
from student_simulator import StudentPopulationSimulator, load_simulation_config

# A shuffled index that is not 0..n-1, so code pairing rows by label fails.
# The spy model reproduces Y exactly only when the exposure it receives is
# the one paired with the rows by position.
_RNG = np.random.default_rng(0)
INDEX = _RNG.permutation(40) + 1000
X = pd.DataFrame(
    {"x": _RNG.normal(size=40), "ignored": _RNG.normal(size=40)}, index=INDEX
)
EXPOSURE = np.arange(1.0, 41.0)
Y = X["x"].to_numpy() * EXPOSURE
N_REPEATS = 4


class _SpyModel(BaseAgeGroupModel):
    """Predicts ``x`` times the exposure; records what each predict call received."""

    def __init__(self) -> None:
        self.predict_exposures: list[ArrayLike | None] = []
        self.rows_in_place: list[bool] = []

    def fit(self, X: Any, y: Any, exposure: ArrayLike | None = None) -> Self:
        return self

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> np.ndarray:
        self.predict_exposures.append(exposure)
        self.rows_in_place.append(bool(np.array_equal(X.index.to_numpy(), INDEX)))
        return np.asarray(X["x"], dtype=float) * np.asarray(exposure, dtype=float)


class _RecordingMetric:
    """The mean absolute error; records each score in call order."""

    def __init__(self) -> None:
        self.scores: list[float] = []

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        score = float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())
        self.scores.append(score)
        return score


def _compute(
    recorder: _RecordingMetric, *, greater_is_better: bool = False, **settings: Any
) -> tuple[_SpyModel, pd.DataFrame]:
    metric = Metric("mae", recorder, greater_is_better=greater_is_better)
    kwargs: dict[str, Any] = {
        "metric": metric,
        "n_repeats": N_REPEATS,
        "random_state": 0,
    }
    model = _SpyModel()
    importance = PermutationImportance(**(kwargs | settings))
    return model, importance.compute_importances(model, X, Y, exposure=EXPOSURE)


def test_an_ignored_column_scores_zero_and_a_used_one_positive() -> None:
    # Catches a scorer that ignores the permuted frame, or the wrong column
    # shuffled: the unused column must score exactly 0 in every repeat.
    _, result = _compute(_RecordingMetric())
    assert result.loc["ignored", "importance_mean"] == 0.0
    assert result.loc["ignored", "importance_std"] == 0.0
    assert result.loc["x", "importance_mean"] > 0.0


def test_a_positive_importance_means_the_column_helps_under_both_directions() -> None:
    # Catches a missing or inverted sign: a loss and a gain must both report
    # the used column as positive.
    _, loss_result = _compute(_RecordingMetric())
    recorder = _RecordingMetric()
    gain = Metric("gain", lambda t, p: -recorder(t, p), greater_is_better=True)
    gain_result = PermutationImportance(
        gain, N_REPEATS, random_state=0
    ).compute_importances(_SpyModel(), X, Y, exposure=EXPOSURE)
    assert loss_result.loc["x", "importance_mean"] > 0.0
    assert gain_result.loc["x", "importance_mean"] > 0.0


def test_exposure_stays_row_aligned_while_columns_are_permuted() -> None:
    # Catches an exposure re-indexed by label, subsampled, or shuffled with
    # the column: the model must see the exposure as given, the rows in
    # place, and reproduce Y exactly before any column is shuffled.
    recorder = _RecordingMetric()
    model, _ = _compute(recorder)
    assert all(exposure is EXPOSURE for exposure in model.predict_exposures)
    assert all(model.rows_in_place)
    assert recorder.scores[0] == 0.0


def test_importance_is_the_mean_and_std_of_the_score_drops() -> None:
    # Catches mean and std swapped, or the permuted score reported instead of
    # the drop. Scores are recorded in order: the baseline, then n_repeats per
    # column. The recorder holds losses and the scorer hands scikit-learn
    # -loss, so the importance is permuted - baseline.
    recorder = _RecordingMetric()
    _, result = _compute(recorder)
    scores = np.array(recorder.scores)
    assert len(scores) == 1 + N_REPEATS * len(X.columns)
    for position, column in enumerate(X.columns):
        start = 1 + position * N_REPEATS
        drops = scores[start : start + N_REPEATS] - scores[0]
        assert result.loc[column, "importance_mean"] == pytest.approx(drops.mean())
        assert result.loc[column, "importance_std"] == pytest.approx(np.std(drops))


def test_same_seed_gives_identical_results() -> None:
    # Catches a seed that is not passed through to the shuffles.
    _, first = _compute(_RecordingMetric())
    _, second = _compute(_RecordingMetric())
    _, other = _compute(_RecordingMetric(), random_state=1)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc["x", "importance_mean"] != other.loc["x", "importance_mean"]


def test_non_finite_score_raises() -> None:
    # Catches NaN importances returned silently: scikit-learn averages a NaN
    # score into the mean without error.
    calls = iter(range(100))
    metric = Metric("one_nan", lambda t, p: float("nan") if next(calls) == 3 else 1.0)
    importance = PermutationImportance(metric, N_REPEATS)
    with pytest.raises(ValueError, match="not finite"):
        importance.compute_importances(_SpyModel(), X, Y, exposure=EXPOSURE)


@pytest.mark.parametrize(
    "settings",
    [
        {"random_stat": 0},
        {"n_repeats": True},
        {"n_repeats": "3"},
        {"metric": {"name": "m", "function": len}},
    ],
    ids=["misspelt-keyword", "bool-repeats", "string-repeats", "metric-dict"],
)
def test_invalid_settings_raise(settings: dict[str, Any]) -> None:
    # Catches lax settings: pydantic's default ignores an extra keyword, reads
    # True as 1 and "3" as 3, and builds a Metric from a dict.
    kwargs: dict[str, Any] = {"metric": Metric("mae", _RecordingMetric())}
    with pytest.raises(ValidationError):
        PermutationImportance(**(kwargs | {"n_repeats": 2} | settings))


def test_total_times_probability_model_with_a_string_column() -> None:
    # Catches a scorer that turns X into an array (the string column and the
    # transformer would break) or that mishandles a DataFrame y.
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    config = load_simulation_config(path)
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(update={"n_neighborhoods": 20})
        }
    )
    table = StudentPopulationSimulator.from_config(config).simulate().dataset
    numeric = ("ses", "avg_household_size", "median_age", "n_daycares_500m")
    features = FeatureTransformer(
        plans=(
            ColumnPlan(name="numeric", columns=numeric, transforms=(Standardize(),)),
            ColumnPlan(
                name="school",
                columns=("school_status",),
                transforms=(
                    OneHot(
                        categories=("none", "existing", "planned"),
                        reference_category="none",
                    ),
                ),
            ),
        )
    )
    X_table = table[[*numeric, "school_status"]]
    Y_table = table[["n_kindergarten", "n_elementary", "n_highschool"]]
    exposure = ExposureTransformer("n_apartments").fit_transform(table)
    model = TotalTimesProbabilityModel(
        total_model=CountModel(
            estimator=PoissonRegressor(),
            use_exposure=True,
            feature_transformer=features,
        ),
        probability_model=CohortProbabilityModel(
            estimator=LogisticRegression(max_iter=1000),
            feature_transformer=features,
        ),
    ).fit(X_table, Y_table, exposure=exposure)
    importance = PermutationImportance(COHORT_POISSON_DEVIANCE, 3, random_state=0)
    result = importance.compute_importances(model, X_table, Y_table, exposure=exposure)
    assert list(result.index) == list(X_table.columns)
    assert result.columns.tolist() == ["importance_mean", "importance_std"]
    assert np.isfinite(result.to_numpy()).all()
