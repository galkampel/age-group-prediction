"""CohortProbabilityModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    Classifier,
    CohortProbabilityModel,
    ReplicationType,
)
from age_group_prediction.scoring import cohort_log_loss

# Not in sorted order, so a mapping by position from the sorted classes_
# ("el", "hs", "kg") would permute the cohorts.
COHORTS = ["kg", "el", "hs"]


def _data(rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multinomial counts whose probabilities move with ``u`` and ``v``, on a non-default index.

    The totals are ``Poisson(4)``, so some buildings have no children; the
    composition is a softmax of known logits (``kg`` the reference).
    """
    rng = np.random.default_rng(seed)
    index = pd.RangeIndex(100, 100 + rows)
    X = pd.DataFrame(
        {"u": rng.normal(size=rows), "v": rng.normal(size=rows)}, index=index
    )
    logits = np.column_stack([np.zeros(rows), 0.5 + 0.8 * X["u"], -0.3 + 0.6 * X["v"]])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    counts = np.vstack(
        [
            rng.multinomial(total, p)
            for total, p in zip(rng.poisson(4, rows), probabilities)
        ]
    )
    return X, pd.DataFrame(counts, columns=COHORTS, index=index)


X, Y = _data(300)


def _logistic() -> LogisticRegression:
    # Unpenalized: the categorical rows then fit the multinomial likelihood exactly.
    return LogisticRegression(C=np.inf, max_iter=1000)


REPLICATIONS: list[ReplicationType] = ["weighted", "per_child"]
CELLS = pd.DataFrame({"kg": [2, 0, 0], "el": [0, 0, 3], "hs": [1, 0, 0]})


def test_the_weighted_rows_are_the_non_zero_cells_with_their_counts() -> None:
    # A zero cell kept as a row, a label from another column, or a weight
    # other than the count would fit another likelihood. The labels are the
    # cohorts' column positions: kg 0, el 1, hs 2.
    positions, labels, weights = CohortProbabilityModel.multinomial_to_categorical(
        CELLS, "weighted"
    )

    np.testing.assert_array_equal(positions, [0, 0, 2])
    np.testing.assert_array_equal(labels, [0, 2, 1])
    np.testing.assert_array_equal(weights, [2, 1, 3])


def test_the_per_child_rows_are_one_unweighted_row_per_child() -> None:
    # A cell not repeated by its count, or weights kept as well, would count
    # each child other than once.
    positions, labels, weights = CohortProbabilityModel.multinomial_to_categorical(
        CELLS, "per_child"
    )

    np.testing.assert_array_equal(positions, [0, 0, 0, 2, 2, 2])
    np.testing.assert_array_equal(labels, [0, 0, 2, 1, 1, 1])
    assert weights is None


def test_an_unknown_replication_raises() -> None:
    # Falling back to one replication would hide a typo or a stale value in a
    # tuned setting.
    with pytest.raises(ValueError, match="got 'replicated'"):
        CohortProbabilityModel.multinomial_to_categorical(CELLS, "replicated")  # type: ignore[arg-type]


def test_the_weighted_fit_equals_a_fit_on_one_row_per_child() -> None:
    # The weight as count/total, no weight, or labels out of step with the
    # rows fits another model than the multinomial likelihood of the counts.
    model = CohortProbabilityModel(estimator=_logistic()).fit(X, Y)
    counts = Y.to_numpy()
    building = np.repeat(np.arange(len(Y)), counts.sum(axis=1))
    labels = np.concatenate([np.repeat(np.arange(len(COHORTS)), row) for row in counts])

    per_child = _logistic().fit(X.iloc[building], labels)

    weighted = model.estimator_
    assert isinstance(weighted, LogisticRegression)
    np.testing.assert_allclose(weighted.coef_, per_child.coef_, atol=1e-10)
    np.testing.assert_allclose(weighted.intercept_, per_child.intercept_, atol=1e-10)


def test_both_replications_fit_the_same_likelihood() -> None:
    # Each child counts once in either: a replication that drops, repeats
    # or reweights children fits another logistic regression.
    weighted = CohortProbabilityModel(estimator=_logistic()).fit(X, Y)
    per_child = CohortProbabilityModel(
        estimator=_logistic(), replication="per_child"
    ).fit(X, Y)

    pd.testing.assert_frame_equal(
        per_child.predict(X), weighted.predict(X), rtol=0, atol=1e-10
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: KNeighborsClassifier(n_neighbors=25),
        # Without metadata routing, which only the weights need.
        lambda: OneVsRestClassifier(LogisticRegression()),
    ],
    ids=["k-neighbors", "one-vs-rest"],
)
def test_per_child_rows_fit_a_classifier_without_sample_weight(
    build: Callable[[], Classifier],
) -> None:
    # Passing sample_weight (even None) would fail for these classifiers.
    model = CohortProbabilityModel(estimator=build(), replication="per_child")

    probabilities = model.fit(X, Y).predict(X)

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)


@pytest.mark.parametrize("with_features", [False, True], ids=["design", "raw-table"])
def test_probabilities_are_named_like_y_and_indexed_like_x(with_features: bool) -> None:
    # Labels that the classifier sorts (the names, not y's column positions)
    # would give a cohort another cohort's probability; misaligned rows would
    # be multiplied by another building's total in Model 2. With a
    # transformer, the index is read from the transformed rows, which must
    # keep the raw table's.
    X_new = X.iloc[:5].set_index(pd.Index([10, 20, 30, 40, 50]))
    features = FeatureTransformer(
        tuple(
            ColumnPlan(name=column, columns=(column,), transforms=(Center(),))
            for column in ("u", "v")
        )
    )
    model = CohortProbabilityModel(
        estimator=_logistic(), feature_transformer=features if with_features else None
    ).fit(X, Y)

    probabilities = model.predict(X_new)

    assert list(probabilities.columns) == COHORTS
    assert probabilities.index.equals(X_new.index)
    design = (
        X_new
        if model.feature_transformer_ is None
        else model.feature_transformer_.transform(X_new)
    )
    np.testing.assert_array_equal(model.estimator_.classes_, [0, 1, 2])
    np.testing.assert_array_equal(
        probabilities.to_numpy(), model.estimator_.predict_proba(design)
    )


CLASSIFIERS: dict[str, Callable[[], Classifier]] = {
    "logistic": _logistic,
    "hist-gradient-boosting": HistGradientBoostingClassifier,
    "lightgbm": lambda: LGBMClassifier(n_estimators=20, n_jobs=1, verbosity=-1),
    "random-forest": lambda: RandomForestClassifier(n_estimators=20, random_state=0),
    # The weights reach the binary classifiers only through metadata routing.
    "one-vs-rest": lambda: OneVsRestClassifier(
        LogisticRegression().set_fit_request(sample_weight=True)
    ),
}


@pytest.mark.parametrize("replication", REPLICATIONS)
@pytest.mark.parametrize("build", CLASSIFIERS.values(), ids=list(CLASSIFIERS))
def test_every_classifier_kind_keeps_the_cohort_order(
    build: Callable[[], Classifier], replication: ReplicationType
) -> None:
    # The protocol must hold for multinomial, tree and one-vs-rest
    # classifiers alike: each must order its classes as y's columns, and a
    # column dropped, repeated or permuted breaks the cohorts or their sum.
    with sklearn.config_context(enable_metadata_routing=True):
        model = CohortProbabilityModel(estimator=build(), replication=replication).fit(
            X, Y
        )
        probabilities = model.predict(X)
        by_class = model.estimator_.predict_proba(X)

    np.testing.assert_array_equal(model.estimator_.classes_, [0, 1, 2])
    np.testing.assert_array_equal(probabilities.to_numpy(), by_class)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)


@pytest.mark.parametrize(
    "names", [["kg", 1, 2.5], [2, 0, 1]], ids=["mixed-types", "integers-unsorted"]
)
def test_cohort_names_of_any_type_name_the_columns_in_order(
    names: list[object],
) -> None:
    # Names as labels would be sorted by the classifier: a str beside a number
    # fails, and unsorted integer names would be permuted silently.
    model = CohortProbabilityModel(estimator=_logistic()).fit(
        X, Y.set_axis(names, axis=1)
    )

    probabilities = model.predict(X)

    assert list(probabilities.columns) == names
    np.testing.assert_array_equal(
        probabilities.to_numpy(), model.estimator_.predict_proba(X)
    )


def test_an_unobserved_cohort_raises_naming_it() -> None:
    # The classifier would never predict it, and predict would lack its column.
    with pytest.raises(ValueError, match=r"no child.*\['hs'\]"):
        CohortProbabilityModel(estimator=_logistic()).fit(X, Y.assign(hs=0))


@pytest.mark.parametrize("replication", REPLICATIONS)
def test_a_building_without_children_adds_nothing_and_is_still_predicted(
    replication: ReplicationType,
) -> None:
    # Its row has no child to label; fitting on it anyway (a zero-weight row
    # or a made-up label) would change the fit.
    empty = Y.sum(axis=1) == 0
    assert empty.any()
    template = CohortProbabilityModel(estimator=_logistic(), replication=replication)

    model = clone(template).fit(X, Y)
    without = clone(template).fit(X[~empty], Y[~empty])

    assert isinstance(model.estimator_, LogisticRegression)
    assert isinstance(without.estimator_, LogisticRegression)
    np.testing.assert_array_equal(model.estimator_.coef_, without.estimator_.coef_)
    assert model.predict(X[empty]).notna().all().all()


def test_a_passed_exposure_is_ignored() -> None:
    # A caller passes one exposure to every model; this one has no exposure.
    exposure = np.arange(1.0, len(X) + 1)
    model = CohortProbabilityModel(estimator=_logistic()).fit(X, Y)
    with_exposure = CohortProbabilityModel(estimator=_logistic()).fit(
        X, Y, exposure=exposure
    )

    pd.testing.assert_frame_equal(
        with_exposure.predict(X, exposure=exposure), model.predict(X)
    )


@pytest.mark.parametrize("replication", REPLICATIONS)
def test_the_feature_transformer_is_fitted_on_the_buildings(
    replication: ReplicationType,
) -> None:
    # Fitted on the categorical rows, its statistics would be weighted by each
    # building's children: here the centre of u moves, since buildings with a
    # large u have more children.
    rng = np.random.default_rng(1)
    X_raw = pd.DataFrame({"u": rng.normal(loc=2.0, size=300)})
    totals = rng.poisson(np.exp(X_raw["u"] - 1))
    y = pd.DataFrame(
        np.vstack([rng.multinomial(t, [0.3, 0.4, 0.3]) for t in totals]),
        columns=COHORTS,
    )
    features = FeatureTransformer(
        (ColumnPlan(name="u", columns=("u",), transforms=(Center(),)),)
    )

    model = CohortProbabilityModel(
        estimator=_logistic(),
        replication=replication,
        feature_transformer=features,
    ).fit(X_raw, y)

    assert model.feature_transformer_ is not None
    pd.testing.assert_frame_equal(
        model.feature_transformer_.transform(X_raw),
        clone(features).fit(X_raw).transform(X_raw),
    )


def test_a_failed_refit_leaves_the_previous_fit_intact() -> None:
    # State set before the classifier's fit succeeds would leave an unfitted
    # classifier, or name the old one's columns after the new cohorts.
    model = CohortProbabilityModel(estimator=_logistic()).fit(X, Y)
    before = model.predict(X)

    with pytest.raises(ValueError, match="NaN"):
        model.fit(X.assign(u=np.nan), Y.set_axis(["a", "b", "c"], axis=1))

    pd.testing.assert_frame_equal(model.predict(X), before)


class _FailingClassifier(LogisticRegression):
    """Fails at ``fit``, after the feature transformer was fitted."""

    def fit(self, *args: object, **kwargs: object) -> _FailingClassifier:
        raise RuntimeError("classifier failed")


def test_a_failed_refit_keeps_the_previous_feature_transformer() -> None:
    # A transformer set before the classifier's fit succeeds would centre new
    # rows on the failed refit's statistics, beside the old classifier.
    features = FeatureTransformer(
        tuple(
            ColumnPlan(name=column, columns=(column,), transforms=(Center(),))
            for column in ("u", "v")
        )
    )
    model = CohortProbabilityModel(
        estimator=_logistic(), feature_transformer=features
    ).fit(X, Y)
    before = model.predict(X)

    with pytest.raises(RuntimeError, match="classifier failed"):
        model.set_params(estimator=_FailingClassifier()).fit(
            X.assign(u=X["u"] + 5.0), Y
        )

    pd.testing.assert_frame_equal(model.predict(X), before)


def test_x_and_y_of_different_lengths_raise() -> None:
    # The rows are taken from y's cells, so a shorter y would fit on X's
    # first rows and drop the rest silently.
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        CohortProbabilityModel(estimator=_logistic()).fit(X, Y.iloc[:-50])


def test_the_template_estimator_stays_unfitted() -> None:
    # Fitting the caller's estimator in place would carry one fold's fit into
    # the next and into the caller's later use.
    template = _logistic()

    model = CohortProbabilityModel(estimator=template).fit(X, Y)

    assert model.estimator_ is not template
    with pytest.raises(NotFittedError):
        check_is_fitted(template)


def test_the_model_beats_the_marginal_shares_on_new_buildings() -> None:
    # A fit that ignores the features, or pairs labels with other buildings'
    # rows, predicts no better than the marginal shares.
    X_all, Y_all = _data(2000, seed=3)
    X_fit, Y_fit, X_new, Y_new = X_all[:1000], Y_all[:1000], X_all[1000:], Y_all[1000:]
    model = CohortProbabilityModel(estimator=_logistic()).fit(X_fit, Y_fit)
    marginal = np.tile(Y_fit.sum() / Y_fit.to_numpy().sum(), (len(X_new), 1))

    model_loss = cohort_log_loss(Y_new.to_numpy(), model.predict(X_new).to_numpy())

    assert model_loss < 0.95 * cohort_log_loss(Y_new.to_numpy(), marginal)


def test_probabilities_that_do_not_sum_to_one_raise() -> None:
    # LightGBM's one-vs-all objective scores each cohort on its own; its rows
    # would reach Model 2 unnormalized, and the cohorts would miss the total.
    model = CohortProbabilityModel(
        estimator=LGBMClassifier(
            objective="multiclassova", n_estimators=20, n_jobs=1, verbosity=-1
        )
    ).fit(X, Y)

    with pytest.raises(ValueError, match="LGBMClassifier's probabilities do not sum"):
        model.predict(X)
