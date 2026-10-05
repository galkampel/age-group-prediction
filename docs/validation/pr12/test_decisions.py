"""PR #12 validation: each decision of the plan's §2 (G1–G6, F1–F3, I1–I2) checked against the code.

Run: uv run pytest -q -c pyproject.toml --rootdir . -W error docs/validation/pr12/
"""

from __future__ import annotations

import inspect
import typing
from typing import Self

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike
from optuna.trial import FixedTrial
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.utils.validation import check_is_fitted

from age_group_prediction import modeling
from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator,
    IntParameter,
)
from age_group_prediction.modeling import (
    CohortProbabilityModel,
    DirectCohortModel,
    IndependentCohortModels,
    IndependentTotalProbabilityModel,
    Regressor,
    TotalChildrenModel,
)
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter


def _lgbm(n: int = 50, objective: str = "poisson") -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        n_estimators=n,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=0,
    )


def _data(
    rows: int = 2000, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"ses": rng.normal(size=rows), "noise": rng.normal(size=rows)})
    exposure = rng.integers(12, 80, size=rows).astype(float)
    y = pd.Series(rng.poisson(exposure * np.exp(-2 + 0.4 * X["ses"])))
    return X, y, exposure


X, Y, E = _data()


# --- G1, G1a ---------------------------------------------------------------


def test_g1_constructor_signature_and_nested_names() -> None:
    params = inspect.signature(DirectCohortModel.__init__).parameters
    assert list(params) == ["self", "estimator", "use_exposure", "feature_transformer"]
    assert all(p.kind is p.KEYWORD_ONLY for name, p in params.items() if name != "self")
    model = DirectCohortModel(estimator=_lgbm(50), use_exposure=True)
    deep = model.get_params(deep=True)
    assert "estimator__n_estimators" in deep
    lightgbm_names = {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "subsample",
        "colsample_bytree",
        "random_state",
        "n_jobs",
        "objective",
    }
    assert not lightgbm_names & set(model.get_params(deep=False))
    trial = clone(model).set_params(estimator__n_estimators=5)
    assert trial.estimator.n_estimators == 5
    assert model.estimator.n_estimators == 50


# --- G1b ---------------------------------------------------------------------


def test_g1b_regressor_is_an_exported_protocol() -> None:
    assert "Regressor" in modeling.__all__
    assert typing.is_protocol(Regressor)
    fit = inspect.signature(Regressor.fit).parameters
    assert list(fit) == ["self", "X", "y", "sample_weight"]
    assert fit["sample_weight"].default is None
    assert list(inspect.signature(Regressor.predict).parameters) == ["self", "X"]


# --- G2 ----------------------------------------------------------------------


def test_g2_estimator_is_required() -> None:
    p = inspect.signature(DirectCohortModel.__init__).parameters["estimator"]
    assert p.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="estimator"):
        DirectCohortModel()  # type: ignore[call-arg]


# --- G3 ----------------------------------------------------------------------


def test_g3a_weighted_rate_equals_lightgbm_init_score_offset() -> None:
    model = DirectCohortModel(estimator=_lgbm(50), use_exposure=True).fit(
        X, Y, exposure=E
    )
    offset = np.log(E) + np.log(Y.sum() / E.sum())
    by_hand = _lgbm(50).fit(X, Y, init_score=offset)
    ours = model.predict(X, exposure=E)
    theirs = np.exp(by_hand.predict(X, raw_score=True) + offset)
    rel = np.max(np.abs(ours - theirs) / theirs)
    print(f"\nG3a max relative difference: {rel:.3e}")
    assert rel < 1e-6


def test_g3b_unpenalized_glm_equals_statsmodels_offset_glm() -> None:
    model = DirectCohortModel(
        estimator=PoissonRegressor(alpha=0.0, max_iter=1000, tol=1e-10),
        use_exposure=True,
    ).fit(X, Y, exposure=E)
    glm = sm.GLM(
        Y, sm.add_constant(X), family=sm.families.Poisson(), offset=np.log(E)
    ).fit()
    ours = np.r_[model.estimator_.intercept_, model.estimator_.coef_]
    print(f"\nG3b ours {ours}, statsmodels {glm.params.to_numpy()}")
    np.testing.assert_allclose(ours, glm.params.to_numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        model.predict(X, exposure=E), glm.fittedvalues, rtol=1e-5
    )


def test_g3c_penalized_glm_alpha_acts_as_alpha_times_mean_exposure() -> None:
    # The plan: the weighted-rate coefficients at alpha match the offset model's at alpha*mean(E).
    alpha = 1.0
    ours = (
        DirectCohortModel(
            estimator=PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-10),
            use_exposure=True,
        )
        .fit(X, Y, exposure=E)
        .estimator_
    )
    # The offset model as a GLM on the count with the same sklearn penalty, by hand:
    # PoissonRegressor has no offset, so fit the rate without weights? No: use the
    # loss directly. Minimize sum_i [E_i e^(b+xB) - y_i (b+xB)] / n + alpha' |B|^2 / 2.
    from scipy.optimize import minimize

    Xv = np.column_stack([np.ones(len(X)), X.to_numpy()])

    def offset_loss(theta: np.ndarray, a: float) -> float:
        eta = Xv @ theta + np.log(E)
        return float(
            np.sum(np.exp(eta) - Y.to_numpy() * eta) / len(Y)
            + a * np.sum(theta[1:] ** 2) / 2
        )

    theirs = minimize(
        offset_loss,
        np.zeros(3),
        args=(alpha * E.mean(),),
        method="BFGS",
        options={"gtol": 1e-10},
    ).x
    coef_ours = np.r_[ours.intercept_, ours.coef_]
    print(f"\nG3c ours {coef_ours}, offset at alpha*mean(E) {theirs}")
    np.testing.assert_allclose(coef_ours, theirs, rtol=1e-3, atol=1e-4)


def test_g3d_hist_gradient_boosting_fits_with_the_exposure() -> None:
    model = DirectCohortModel(
        estimator=HistGradientBoostingRegressor(
            loss="poisson", max_iter=50, random_state=0
        ),
        use_exposure=True,
    ).fit(X, Y, exposure=E)
    pred = model.predict(X, exposure=E)
    assert pred.shape == (len(X),) and np.all(pred > 0)
    assert pred.mean() == pytest.approx(Y.mean(), rel=0.01)


# --- G4 ----------------------------------------------------------------------


def test_g4_gaussian_weighted_rate_is_wls_of_the_count() -> None:
    model = DirectCohortModel(estimator=LinearRegression(), use_exposure=True).fit(
        X, Y, exposure=E
    )
    design = np.column_stack([E, E[:, None] * X.to_numpy()])
    wls = sm.WLS(Y, design, weights=1 / E).fit()
    np.testing.assert_allclose(
        model.predict(X, exposure=E), wls.fittedvalues, rtol=1e-10
    )
    lgbm = DirectCohortModel(estimator=_lgbm(20, "regression"), use_exposure=True).fit(
        X, Y, exposure=E
    )
    np.testing.assert_allclose(
        lgbm.predict(X, exposure=2 * E), 2 * lgbm.predict(X, exposure=E)
    )


# --- G5 ----------------------------------------------------------------------


class _NoWeight(BaseEstimator):
    def fit(self, X: pd.DataFrame, y: ArrayLike) -> Self:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))


def test_g5_no_pre_check_of_sample_weight() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'sample_weight'"):
        DirectCohortModel(estimator=_NoWeight(), use_exposure=True).fit(
            X, Y, exposure=E
        )
    source = inspect.getsource(DirectCohortModel.fit)
    assert "sample_weight" not in source.replace("sample_weight=exposure_values", "")
    # Without the exposure the same estimator is fine (no check at all).
    DirectCohortModel(estimator=_NoWeight()).fit(X, Y)


# --- G6 ----------------------------------------------------------------------


def test_g6_fitted_state_and_predict_follows_it() -> None:
    model = DirectCohortModel(estimator=_lgbm(5), use_exposure=True).fit(
        X, Y, exposure=E
    )
    fitted = {k for k in vars(model) if k.endswith("_")}
    assert fitted == {"estimator_", "use_exposure_", "feature_transformer_"}
    assert not hasattr(model, "regressor_") and not hasattr(model, "base_log_rate_")
    before = model.predict(X, exposure=E)
    model.set_params(use_exposure=False)
    with pytest.raises(ValueError, match="pass `exposure`"):
        model.predict(X)
    np.testing.assert_array_equal(model.predict(X, exposure=E), before)
    np.testing.assert_allclose(model.predict(X, exposure=2 * E), 2 * before)


# --- F1 ----------------------------------------------------------------------


def test_f1_feature_transformer_on_leaf_models_only_and_helpers() -> None:
    for cls in (DirectCohortModel, TotalChildrenModel, CohortProbabilityModel):
        assert "feature_transformer" in inspect.signature(cls.__init__).parameters, cls
    for cls in (IndependentCohortModels, IndependentTotalProbabilityModel):
        assert (
            "feature_transformer" not in inspect.signature(cls.__init__).parameters
        ), cls
    tree = FeatureTransformer(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    table = pd.DataFrame({"x": [1.0, 2.0, 6.0], "drop": [0.0, 0.0, 0.0]})
    model = DirectCohortModel(estimator=_lgbm(2), feature_transformer=tree).fit(
        table, pd.Series([1, 2, 3])
    )
    assert model.feature_transformer_ is not tree
    with pytest.raises(NotFittedError):
        check_is_fitted(tree)
    new = pd.DataFrame({"x": [0.0, 10.0], "drop": [1.0, 1.0]})
    design = model._transform_features(new)
    np.testing.assert_allclose(
        design["x"], new["x"] - 3.0
    )  # training mean, not the new rows'
    assert list(design.columns) == ["x"]


# --- F2 ----------------------------------------------------------------------


def test_f2_model_pipeline_is_gone() -> None:
    assert "ModelPipeline" not in modeling.__all__ and not hasattr(
        modeling, "ModelPipeline"
    )
    assert "Objective" not in modeling.__all__
    with pytest.raises(ModuleNotFoundError):
        import age_group_prediction.modeling.pipeline  # noqa: F401


# --- F3 ----------------------------------------------------------------------


class _RecordingTree(FeatureTransformer):
    """Records the row ids seen by each fit and the fitted-mean used at each transform."""

    fits: typing.ClassVar[list[list[int]]] = []
    transforms: typing.ClassVar[list[tuple[float, list[int]]]] = []

    def fit(self, X: pd.DataFrame, y: object = None) -> _RecordingTree:
        _RecordingTree.fits.append(sorted(X["id"].astype(int).tolist()))
        self.mean_id_ = float(X["id"].mean())
        return super().fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _RecordingTree.transforms.append(
            (self.mean_id_, sorted(X["id"].astype(int).tolist()))
        )
        return super().transform(X)


def test_f3_evaluator_has_build_model_and_fits_the_model_per_fold_on_raw_rows() -> None:
    fields = {f.name for f in CVHyperparameterEvaluator.__dataclass_fields__.values()}
    assert "feature_transformer" not in fields
    assert hasattr(CVHyperparameterEvaluator, "build_model")
    assert not hasattr(CVHyperparameterEvaluator, "build_feature_transformer_and_model")

    rng = np.random.default_rng(1)
    n = 60
    table = pd.DataFrame({"id": np.arange(n, dtype=float), "x": rng.normal(size=n)})
    y = pd.Series(rng.poisson(np.exp(1 + 0.3 * table["x"])))
    groups = np.repeat(np.arange(6), 10)
    tree = _RecordingTree(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    evaluator = CVHyperparameterEvaluator(
        DirectCohortModel(estimator=_lgbm(5), feature_transformer=tree),
        [IntParameter("estimator__n_estimators", 2, 10)],
        cv=Splitter("grouped").cv(n_splits=3, random_state=0),
        metric=POISSON_DEVIANCE,
    )
    _RecordingTree.fits.clear()
    _RecordingTree.transforms.clear()
    evaluator.evaluate(FixedTrial({"estimator__n_estimators": 3}), table, y, groups)
    folds = list(
        Splitter("grouped").cv(n_splits=3, random_state=0).split(table, y, groups)
    )
    assert len(_RecordingTree.fits) == 3
    for k, (train, val) in enumerate(folds):
        assert _RecordingTree.fits[k] == sorted(train.tolist())
        fit_mean = float(train.mean())
        # Two transforms per fold: the training design at fit, the validation rows at predict.
        assert _RecordingTree.transforms[2 * k] == (fit_mean, sorted(train.tolist()))
        assert _RecordingTree.transforms[2 * k + 1] == (fit_mean, sorted(val.tolist()))
    with pytest.raises(NotFittedError):
        check_is_fitted(evaluator.model.feature_transformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(evaluator.model)


# --- I1, I2 --------------------------------------------------------------------


def test_i1_i2_prediction_follows_y_order_and_the_template_stays_unfitted() -> None:
    rng = np.random.default_rng(2)
    n = 100
    table = pd.DataFrame({"x": rng.normal(size=n)})
    y = pd.DataFrame({"a": rng.poisson(2, n), "b": rng.poisson(3, n)})
    templates = {
        "b": DirectCohortModel(estimator=_lgbm(3)),
        "a": DirectCohortModel(estimator=_lgbm(3)),
    }
    model = IndependentCohortModels(templates).fit(table, y)
    pred = model.predict(table)
    assert list(pred.columns) == ["a", "b"]
    assert list(model.cohort_models_) == ["a", "b"]
    assert model.cohort_models is templates
    for cohort, template in templates.items():
        with pytest.raises(NotFittedError):
            check_is_fitted(template)
        assert model.cohort_models_[cohort] is not template
        check_is_fitted(model.cohort_models_[cohort])
