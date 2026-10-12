# Total Times Probability Model (Model 2)

> The rebuilt Model B / Model 2 in `age_group_prediction.modeling`:
> `TotalTimesProbabilityModel`, the predicted total (`CountModel`) times a classifier's
> cohort probabilities (`CohortProbabilityModel`). Built and tested; nothing calls it yet.
> The original `models/independent_total_probability.py`, which `experiment/` and
> `tracking/` still run, is described in
> [INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md) §1–§12
> until the old stack is deleted ([roadmap](MODEL_REIMPLEMENTATION_PLAN.md)). No torch,
> no hand-written objective: scikit-learn, LightGBM and statsmodels estimators only.
> Decisions, measured evidence and the build's record:
> [TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md) (§5
> P1–P15, §6, §8).

Each component below: what it is, the formula, the derivation, why, the code, the
measured check. `CountModel`, shared with Model 1, is explained in
[Independent cohort models §2](INDEPENDENT_COHORT_MODELS.md#2-countmodel).

## 1. The Model

For building $b$ with apartment count $A_b$ (the exposure), feature vectors $x_b$ (total
model) and $w_b$ (probability model), cohort counts $C_{b,k}$, $k = 1, \dots, K$, and total
$Y_b = \sum_k C_{b,k}$:

**Total** (`CountModel`): a regression of $Y_b$ on $x_b$ whose mean is proportional to the
exposure,

$$
\mu_b = A_b\, f(x_b),
\qquad
\text{for a log-link GLM}\quad \log\mu_b = \log A_b + \beta_0 + x_b^\top\beta,
$$

with $Y_b \sim \operatorname{Poisson}(\mu_b)$ or $Y_b \sim \operatorname{NB2}(\mu_b,
\alpha)$, $\operatorname{Var}(Y_b) = \mu_b(1 + \alpha\mu_b)$.

**Probabilities** (`CohortProbabilityModel`): conditional on the total, the children of a
building are a multinomial draw,

$$
(C_{b,1}, \dots, C_{b,K}) \mid Y_b \sim \operatorname{Mult}(Y_b,\ p_b),
\qquad
p_{b,k} = P(\text{cohort } k \mid w_b),
$$

and $p_b$ is a multi-class classifier's `predict_proba` on $w_b$, fitted on the children
as categorical rows (§3.1): multinomial logistic regression $p_{b,k} =
\operatorname{softmax}(a_k + w_b^\top\gamma_k)$, or a tree ensemble.

**Combined** (`TotalTimesProbabilityModel`):

$$
\widehat C_{b,k} = \widehat\mu_b\,\widehat p_{b,k},
\qquad
\sum_k \widehat C_{b,k} = \widehat\mu_b\sum_k\widehat p_{b,k} = \widehat\mu_b,
$$

since every classifier's rows sum to 1 (`predict` raises if one does not, §3.2): the
cohort predictions reconcile with the total exactly. The halves share no parameter, so
`fit` fits them independently, the total model on `y.sum(axis=1)` and the probability
model on `y`, both given the same `exposure` (the probability model ignores it: shares
do not depend on building size).

## 2. The Total: `CountModel`

The total is the same `CountModel` as Model 1's, given the row sum of `y`
([Independent cohort models §2](INDEPENDENT_COHORT_MODELS.md#2-countmodel)): any
regressor, the exposure as the weighted per-apartment rate for an estimator with
`sample_weight` (exactly the Poisson offset model) or passed raw to one whose `fit` takes
`exposure` (`NegativeBinomialRegressor`, statsmodels' NB2, for which the rate form is a
different model). What is specific here: `TotalChildrenModel` is gone (one class serves
a cohort and the total); the penalty is the estimator's own, and under the rate form
`PoissonRegressor(alpha)` acts as `alpha × mean(exposure)` of the offset model, so
sklearn's default `alpha=1` is a penalized total (§9: it no longer overfits the
population PR #11's unpenalized torch Poisson did); NB2 is unpenalized.

## 3. `CohortProbabilityModel`

`estimator` is any scikit-learn multi-class classifier (required); `replication` builds
the categorical rows (§3.1); `calibration_method` / `calibration_cv` calibrate inside the
model (§3.4); `feature_transformer` as in `CountModel`. `fit(X, y)` takes the raw count
table; `predict` returns the probabilities, a DataFrame with `y`'s columns.

### 3.1 The categorical rows

**What.** `fit` turns the count table into categorical rows itself
(`multinomial_to_categorical(y, replication)`): `"weighted"` (default), one row per
(building, cohort) with a child, `sample_weight` = the count; `"per_child"`, one row per
child, no `sample_weight`. A row's label is its cohort's **column position** in `y`.

**Derivation (the likelihood identity).** Under §1's multinomial model the
log-likelihood of building $b$ is

$$
\log P(C_b \mid Y_b, p_b) = \log\frac{Y_b!}{\prod_k C_{b,k}!} + \sum_k C_{b,k}\log p_{b,k},
$$

and the first term is free of $p_b$. The second is the log-likelihood of $Y_b$
independent categorical draws, $C_{b,k}$ of them in cohort $k$, each with features $w_b$:
one row per child. For an estimator that maximizes a **weighted** log-likelihood, a
`sample_weight` of $c$ multiplies a row's term by $c$, exactly what $c$ identical rows
contribute, so one row per positive cell with weight $C_{b,k}$ is the same objective:

$$
\sum_b \sum_{k:\,C_{b,k} > 0} C_{b,k}\log p_k(w_b)
= \sum_b \sum_{\text{child } i \text{ of } b} \log p_{k(i)}(w_b).
$$

It is also what `COHORT_LOG_LOSS` scores (per child; "equals scikit-learn's `log_loss`
on one row per child"). A weight of $C_{b,k}/Y_b$ instead (one unit per building, the
Dirichlet build's weighting) gives $\sum_b\sum_k s_{b,k}\log p_k(w_b)$: not the
multinomial likelihood, and a different estimator (coefficients differ by 0.059).

**Why two replications.** For a weighted-likelihood estimator (LR, HGB, LightGBM) they
are the same fit. For an estimator that counts or resamples **rows** they are different
procedures: `min_samples_leaf` / `min_child_samples` count rows, not weights, and a
bootstrap draws rows (§3.3). Measured on the simulator (five populations, held-out
`COHORT_LOG_LOSS`, per child / weighted; the per-seed difference; fit seconds): LR 1.0783
/ 1.0783 (0; 0.005 / 0.003 s); HGB 1.0899 / 1.0869 (−0.001 to +0.010; 3.7 / 1.3); LightGBM
1.0884 / 1.0850 (−0.002 to +0.006; 0.12 / 0.04); RF 300 trees 1.0822 / 1.0817 (−0.001 to
+0.005; 0.38 / 0.19). Within noise, so the default is the cheaper `"weighted"`;
`"per_child"` is 2–3× slower (rows × children), needs counts of an integer dtype numpy
casts safely to `int64` (`np.repeat` rejects floats, `uint64`, pandas' `Int64`), and
**admits classifiers without `sample_weight`** (KNN, LDA, QDA, Gaussian process, nearest
centroid, label propagation, `OneVsRestClassifier` without metadata routing), since no
keyword is passed. Under `"weighted"` such a classifier fails in the library
(`TypeError` or `ValueError`; `BaggingClassifier` would use the weights as sampling
probabilities), left so: an up-front check would refuse `OneVsRestClassifier` with
routing on.

**Why inside `fit`.** Three things need the original rows: the feature transformer is
fitted on the **buildings** first (a fit on the replicated rows would weight its
statistics by children; `test_the_feature_transformer_is_fitted_on_the_buildings`); the
calibration folds split the buildings (§3.4); and the base contract keeps `y` as counts,
so the probability model is fitted, scored and tuned on its own and
`TotalTimesProbabilityModel` only passes `y` on.

**Why positions as labels.** Every classifier sorts its labels into `classes_`
(`['el', 'hs', 'kg']` for `['kg', 'el', 'hs']`), which `predict` would have to map back,
and names of mixed types fail the sort. With positions `classes_` is `0 … K−1` in `y`'s
order (measured for LR, HGB, LightGBM, RF, OvR, KNN, `CalibratedClassifierCV`, $K = 2$,
integer names colliding with positions), so `predict` is
`DataFrame(predict_proba(X), columns=cohorts_, index=X.index)`.

**Edge cases.** A building without children adds no row (it still trains the total). A
cohort with no child in `y` raises at `fit`, naming it: the classifier cannot learn an
absent class and `predict` would lack its column. `check_consistent_length(X, y)`: the
rows come from `y`'s cells, so a shorter `y` would fit on `X`'s first rows silently.
Count values (non-negative, finite) are validated where the data is prepared, not here:
a negative or NaN count is a negative or NaN weight, which LR, HGB and LightGBM fit
silently ([plan §10](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md), a follow-up validator).

**Check.** Unpenalized `LogisticRegression` on the weighted rows vs one row per child:
coefficients within $1.8 \times 10^{-15}$ (442 vs 1,325 rows, 2026-10-07) and $6.1 \times
10^{-16}$ (the class's two replications, 2026-10-08).

### 3.2 The classifier

**The protocol.** `Classifier`: `fit(X, y, sample_weight=None)`, `predict_proba(X)`,
`classes_`. Softmax (multinomial) losses: `LogisticRegression` (lbfgs is multinomial in
scikit-learn 1.9; unpenalized is `C=np.inf`, `penalty=None` raises a `FutureWarning`),
`HistGradientBoostingClassifier`, `GradientBoostingClassifier`, `MLPClassifier`, LightGBM
`"multiclass"` (its default above 2 classes), CatBoost `"MultiClass"`, XGBoost
`multi:softprob`. Every other common classifier is multiclass by construction (trees,
naive Bayes, neighbors, LDA/QDA) or reduces to binary internally (`SGDClassifier`,
`GaussianProcessClassifier`, `SVC`); the binary-only ones raise (the threshold
meta-estimators; LightGBM `"binary"`, `"cross_entropy"`), and those without
`predict_proba` (`LinearSVC`, `SVC` without `probability=True`, `RidgeClassifier`, …)
fail at `predict`.

**Why no detection or wrap.** scikit-learn's `classifier_tags.multi_class` is `True` for
every classifier (LightGBM `"binary"` too), so the model cannot tell, and no common
classifier lacks a multiclass fit; a caller who wants one-vs-rest wraps in
`OneVsRestClassifier` explicitly. `OneVsRestClassifier.fit(X, y, **fit_params)` takes
`sample_weight` only through metadata routing (`sklearn.set_config(enable_metadata_routing=True)`,
`set_fit_request(sample_weight=True)` on the inner estimator; measured: the weights reach
it, the fit moves by 0.065); under `"per_child"` no keyword is passed, so it fits without
routing.

**Rows must sum to 1.** `predict` raises if a row of `predict_proba` does not (atol
$10^{-6}$; softmax classifiers measured within $2.2 \times 10^{-16}$, `OneVsRestClassifier`
normalizes in the multiclass case): LightGBM's `objective="multiclassova"` scores each
cohort alone and returns rows summing to 0.67–1.28 silently, and Model 2 would then
predict cohorts that miss its total (a calibration method renormalizes it, so the check
fires only without one). The error shows the row sums' range, so NaN probabilities are
not mistaken for a one-vs-all objective. **Code:** `_check_rows_sum_to_one`.

### 3.3 Bagging and bootstrap under replication

**The question.** A building's cohorts are connected: they share $w_b$, they sum to
$Y_b$, and their probabilities sum to one. Should a resampling estimator (a forest's
bootstrap, LightGBM's bagging) keep a building's rows together?

**Derivation.** (a) The sum to one is a constraint on the model's **output** $p_b$,
enforced by every `predict_proba` (softmax or normalization) and by
`CalibratedClassifierCV`; it is not a dependence between rows. The shared features and
the shared total are properties of the rows. (b) Under §1's conditional model $C_b \mid
Y_b \sim \operatorname{Mult}(Y_b, p_b)$ is $Y_b$ independent categorical draws given
$w_b$: the children of a building are conditionally i.i.d., so the exchangeable unit is
the **child**, and the weighted representation merely compresses identical child rows
into one cell per cohort. (c) Under `"weighted"` rows a bootstrap draws **cells** (a bag
can hold $b$'s kindergarten cell and drop its elementary cell; with `sample_weight`,
scikit-learn multiplies the weight by the draw count); under `"per_child"` rows it draws
**children** (a bag thins each building's composition at random). Neither keeps a
building whole; only a bootstrap **by building** does, which scikit-learn and LightGBM do
not offer. All three are unbiased for the composition: for every unit $i$ the draw count
$m_i$ has $E[m_i] = 1$, so $E[\sum_i m_i \ell_i] = \sum_i \ell_i$; what differs is each
bag's variance, and a bag by building drops whole buildings too.

**What the libraries do** (scikit-learn 1.9, LightGBM 4.7). `RandomForestClassifier`
bootstraps with replacement, **on by default**; LightGBM subsamples without replacement
only with `subsample_freq > 0`; HGB, `LogisticRegression` and XGBoost (`subsample=1.0`)
resample nothing by default. So the forest is the only common estimator that does.

**Check.** RF, 300 trees, `min_samples_leaf=5`, 5 seeds, 2,000 test buildings, held-out
cross-entropy against the **true** $p$: weighted + bootstrap 0.712, per-child +
bootstrap 0.708, weighted `bootstrap=False` 0.716, per-child `bootstrap=False` 0.717, a
hand-made bootstrap by building 0.717; seed spread 0.57–0.86. The forests differ
pointwise (up to 0.26) but none is better: all within noise.

**Why nothing is built.** Observed building features (type, year) change nothing: every
row of a building already shares its whole $w_b$, and conditioning on more of it makes
its children's conditional independence more plausible. The building becomes the right
unit only for an **unobserved** building effect (a random building intercept in the
generator, or a characteristic withheld from the model: extra-multinomial variation) or
for uncertainty intervals. If real data shows that: `bootstrap=False` on the forest (its
randomness then comes from `max_features`), or a bootstrap-by-building aggregator over
`CohortProbabilityModel`, a ten-line class taking `groups` by `has_fit_parameter` as the
NB2 exposure does, added only then. What to tune meanwhile: the forest's
`min_samples_leaf` and `max_features`, LightGBM's `subsample` / `subsample_freq` and
`min_child_samples`, and `replication` as a setting (`probability_model__replication`).
The folds that **must** be grouped are the calibration folds (by building, §3.4) and the
tuner's (by neighborhood, `Splitter`).

### 3.4 Calibration

**What.** `calibration_method` (`None` default; `"temperature"`, `"sigmoid"`,
`"isotonic"`) wraps the cloned classifier in scikit-learn's
`CalibratedClassifierCV(estimator, method=…, cv=…, ensemble=False)` inside `fit`, with
`calibration_cv` (5) folds of **buildings**.

**Cross-fitting, not an ensemble.** With `ensemble=False` the classifier is fitted on
each of $k$ training folds, its out-of-fold scores for **every** row are collected,
**one** calibrator is fitted on them, and the classifier is refitted on all rows;
`predict` is that one model through that one map. With `ensemble=True` (sklearn's
default for a non-frozen estimator) $k$ calibrated fold models are averaged and none is
fitted on all rows. Cross-fitting is what PR #11's build did by hand and keeps one final
model; inside the model, `clone`, `set_params` (`probability_model__calibration_method`)
and the tuner carry it, with no `FrozenEstimator` idiom. Measured: one calibrated model
(`len(calibrated_classifiers_) == 1`), `classes_ == [0, 1, 2]`, rows summing to 1 within
$2.2 \times 10^{-16}$, for LR, LightGBM and RF under all three methods.

**Folds of buildings, split before the categorical rows.** A building's rows carry its
known composition: with plain `KFold` over the rows (or `cv=int`, which stratifies by
class) the same building sits in a fit fold and in its calibration fold, and the
calibrator sees in-sample confidence. So the **buildings** are split, round-robin
(building $i$ in fold $i \bmod k$: deterministic, each fold spanning the table even if it
is sorted), and each categorical row goes to its building's fold:
`PredefinedSplit(sample_folds[sample_positions])`. The split is made before the
replication so that both replications calibrate on the **same** folds: `GroupKFold` over
the rows balanced cells under `"weighted"` and children under `"per_child"`, different
partitions, and the fitted inverse temperature differed (`beta_` 0.9628 vs 0.9695 for the
same unpenalized LR); split first, both give 0.968229 ($2.2 \times 10^{-13}$).
Limitation: the repo's evaluation unit is the neighborhood, but the base contract carries
no `groups`, so the calibration folds are by building; the tuner's outer folds are by
neighborhood regardless. No check on `calibration_cv`: 1, 0 and negative raise in
sklearn; more folds than buildings gives one per building, slower, not wrong.

**Five folds.** The calibrator must map the **final** model's scores but is fitted on
fold models trained on $(k-1)/k$ of the rows, which are less confident than the final
one: $k = 2$ calibrates a model fitted on half the data and biases the temperature toward
sharpening; larger $k$ approaches the final model at the cost of $k$ fits. With 179–206
training buildings (the smoke run's splits), 5 (sklearn's default) leaves 143–165 per
fold fit and uses every row for the calibrator.

**The three methods.** Temperature: one parameter $\beta$ (the inverse temperature,
sklearn's `beta_`), $p = \operatorname{softmax}(\beta\ell)$ on the logits, a power
transform of the probabilities. Sigmoid (Platt): 2 parameters per class on a one-vs-rest
score, then renormalized. Isotonic: a non-parametric monotone map per class,
renormalized; scikit-learn advises it only well above ~1,000 samples per class. Every row
of a 5-fold cross-fit (on the simulator a training set holds ~4,200 children in ~570
cells, 1,200–1,600 per cohort) is ample for 1 or $2K$ parameters and at isotonic's
threshold, where it is the worst method for LR below. All three return rows summing to 1
($2.2 \times 10^{-16}$). The calibrator sees `decision_function` when the estimator has
one (LR, LightGBM, HGB: logits) and `predict_proba` otherwise (RF, KNN). The weights reach
both the fold fits and the calibrator (`beta_` 0.9655 with the counts as weights vs
0.9615 without).

**Check** (held-out `COHORT_LOG_LOSS`, ten populations, grouped 80/20, four standardized
features, test buildings with a child; mean ± sd; sub-task 4):

| Estimator | None | temperature | sigmoid | isotonic |
|---|---|---|---|---|
| `LogisticRegression(C=inf)` | 1.0831 ± 0.0098 | 1.0827 ± 0.0090 (`beta_` 0.89) | 1.0829 ± 0.0092 | 1.0849 ± 0.0090 |
| `LGBMClassifier()` defaults | 1.0888 ± 0.0104 | 1.0876 ± 0.0062 (`beta_` 0.48) | 1.0868 ± 0.0085 | 1.0874 ± 0.0082 |

Every method is within noise of `None` for these two (seed sd ≈ 0.01). At the class
defaults (§9, on other features) the tree classifiers are overconfident (`beta_`
0.39–0.55, $T \approx 1.8$–2.6; LR 0.82) and temperature repairs most of it. Fit cost
×1.6 (LR) to ×5 (HGB: 6 fits).

**Documented, not checked.** A training fold lacking a cohort (a cohort seen in very few
buildings): a classifier with `decision_function` raises ("Only 2 class/es in training
fold, but 3 in overall dataset"), one with `predict_proba` alone (RF, KNN) warns and the
calibrator sees that cohort at 0 out of fold; on the simulator every cohort is in 174+ of
~180–206 training buildings and no fold lacked one (0 of 150 folds, $k$ = 2, 5, 10).
Under a method, `"weighted"` with a classifier lacking `sample_weight` only **warns** and
fits it unweighted ("sample weights will only be used for the calibration itself"),
where without a method it raises: such a classifier (KNN) goes under `"per_child"`. With
metadata routing enabled, every calibrated classifier needs
`set_fit_request(sample_weight=True)` (`UnsetMetadataPassedError`, loud).

**Code.** `CohortProbabilityModel.fit`: `sample_folds = arange(len(y)) % calibration_cv`,
`PredefinedSplit(sample_folds[sample_positions])`, `ensemble=False`; `CalibrationMethod`.

## 4. `TotalTimesProbabilityModel`

**What.** `TotalTimesProbabilityModel(total_model, probability_model)`: `fit` clones both,
fits `total_model` on `y.sum(axis=1)` and `probability_model` on `y`, both with
`exposure`, and sets `total_model_` / `probability_model_` only once both succeeded (a
failed refit leaves the previous fit intact). `predict` is

```python
total = np.asarray(self.total_model_.predict(X, exposure=exposure), dtype=float)
return self.probability_model_.predict(X, exposure=exposure) * total[:, None]
```

**Why this shape.** `probability_model` is typed `CohortProbabilityModel`, whose
`predict` already returns `y`'s columns at fit on `X`'s index, so the product keeps them
and no column check is needed (the planned check could fire only for another probability
model reordering its columns, which does not exist). The total goes through `np.asarray`:
a Series total with its own index would be realigned to `X`'s, into NaN, silently. The
numpy broadcast `* total[:, None]` is along the rows (a bare `* total` would align with
the **columns**: an error at 60 rows, silent when rows equal cohorts); it is the same as
`mul(total, axis=0)` and faster (31 vs 56 µs on 245 rows). Nested names reach both halves
(`total_model__estimator__alpha`, `total_model__use_exposure`,
`probability_model__estimator__C`, `probability_model__replication`,
`probability_model__calibration_method`); the templates stay unfitted, so a refit equals
a fresh fit (the contract test). **Check.** The rows of every smoke-run prediction equal
the total model's prediction (asserted on all 240 fits).

## 5. API

Import from `age_group_prediction.modeling`; the package root exports the **original**
classes of the old stack. `CountModel` and `NegativeBinomialRegressor`:
[Independent cohort models §4](INDEPENDENT_COHORT_MODELS.md#4-api).

| Class | Settings (defaults) | `fit` takes | Fitted state | `predict` returns |
|---|---|---|---|---|
| `CohortProbabilityModel` | `estimator` (required: a `Classifier`); `replication="weighted"` or `"per_child"`; `calibration_method=None`, `"temperature"`, `"sigmoid"` or `"isotonic"`; `calibration_cv=5`; `feature_transformer=None` | the raw table with `feature_transformer`, otherwise the design matrix; a DataFrame of cohort counts (≥ 2 columns, every cohort with a child, counts non-negative and finite, integer under `"per_child"`); an exposure is ignored | `estimator_` (the fitted classifier, or the `CalibratedClassifierCV` around it), `cohorts_` (`y`'s columns), `feature_transformer_` | the probabilities, a DataFrame with `y`'s columns, indexed like `X`, rows summing to 1 |
| `TotalTimesProbabilityModel` | `total_model` (a `BaseAgeGroupModel`, e.g. `CountModel`), `probability_model` (a `CohortProbabilityModel`) | the raw table, the cohort counts, `exposure=` for both models | `total_model_`, `probability_model_` | `total × probabilities`, a DataFrame with `y`'s columns at fit, indexed like `X`, each row summing to the predicted total |

Every model keeps `fit(X, y, exposure=None)`, `predict(X, exposure=None)` and
`evaluate(y_true, y_pred, metric)`. Type aliases: `Classifier`, `ReplicationType`,
`CalibrationMethod`.

## 6. Data Flow

1–2. Preprocessing on the full table and the split by neighborhood, as
   [Independent cohort models §5](INDEPENDENT_COHORT_MODELS.md#5-data-flow): rows are
   paired by position (nothing checks the index), and the targets come from `y` only,
   so `X` may keep the target columns under the transformers' `remainder="drop"`.
3. **Fit on the training rows.** Each half fits its own feature transformer on those
   rows; the calibration folds, if any, are made inside the probability model's `fit`
   from the training buildings only.

```python
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from age_group_prediction.modeling import (
    CohortProbabilityModel, CountModel, NegativeBinomialRegressor, TotalTimesProbabilityModel,
)
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import COHORT_LOG_LOSS, POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import take_rows

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]

# 1. On the full table: raw_table is the simulator's or the real data
table = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(raw_table)
exposure = ExposureTransformer("n_apartments").fit_transform(table)

# 2. The split: row positions, applied to every array alike
train_index, test_index = Splitter("grouped").train_test_indices(
    table, table["neighborhood_id"], test_size=0.2, random_state=0
)
X_train, X_test = take_rows(table, train_index), take_rows(table, test_index)
Y_train, Y_test = take_rows(table[COHORTS], train_index), take_rows(table[COHORTS], test_index)
exposure_train, exposure_test = take_rows(exposure, train_index), take_rows(exposure, test_index)

# 3. Model 2: the exposure goes to both halves; the probability model ignores it.
#    total_base and cohort_probability_base are Feature transformations §8.2–§8.3
model = TotalTimesProbabilityModel(
    total_model=CountModel(
        estimator=NegativeBinomialRegressor(),  # or PoissonRegressor(alpha=1e-3), LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)
        use_exposure=True,
        feature_transformer=total_base,
    ),
    probability_model=CohortProbabilityModel(
        estimator=LogisticRegression(),  # or LGBMClassifier(n_jobs=1, verbosity=-1), RandomForestClassifier(n_jobs=1)
        calibration_method="temperature",  # None: the classifier's own probabilities
        feature_transformer=cohort_probability_base,
    ),
).fit(X_train, Y_train, exposure=exposure_train)
predictions = model.predict(X_test, exposure=exposure_test)  # a DataFrame, total × probabilities
scores = {
    cohort: model.evaluate(Y_test[cohort], predictions[cohort], POISSON_DEVIANCE)
    for cohort in COHORTS
}
scores["total"] = model.evaluate(Y_test.sum(axis=1), predictions.sum(axis=1), POISSON_DEVIANCE)
scores["composition"] = model.evaluate(Y_test, predictions, COHORT_LOG_LOSS)  # per child
alpha = model.total_model_.estimator_.dispersion_  # NB2's fitted α
```

Run as written on the simulator's seed-0 population (2026-10-08, under `-W error`): 196
training and 49 test buildings; `scores` 1.855 (kindergarten), 1.958 (elementary), 2.250
(high school), total 3.587, composition 1.084; $\alpha$ 0.0876;
`predictions.sum(axis=1)` equals the total model's prediction.

## 7. Errors

| Raises | When |
|---|---|
| the total model's | [Independent cohort models §6](INDEPENDENT_COHORT_MODELS.md#6-errors): the exposure checks, `TypeError` for an estimator taking neither argument, NB2's `RuntimeError` |
| `ValueError` "inconsistent numbers of samples" | `X` and `y` of different lengths at `CohortProbabilityModel.fit` |
| `ValueError` "no child is observed in cohorts" | a cohort column of `y` sums to 0 |
| `ValueError` "replication must be" | an unknown `replication` |
| `ValueError` "probabilities do not sum to 1" | a one-vs-all objective without a calibration method (§3.2) |
| `ValueError` "Only 2 class/es in training fold" | a calibration fold lacking a cohort, for a classifier with `decision_function` (§3.4) |
| `InvalidParameterError` / `ValueError` from scikit-learn | an unknown `calibration_method`; `calibration_cv` of 1, 0 or negative |
| `TypeError` / `ValueError` from the library | a classifier without `sample_weight` under `"weighted"` and no calibration method; non-integer counts under `"per_child"` (`np.repeat`) |
| `NotFittedError` | `predict` before `fit` |

Left to the libraries or to preprocessing: count values, the exposure's values
(`ExposureTransformer`), NaN in `X`, reordered columns without a feature transformer
(LightGBM accepts them silently; the transformer selects by name; plan §10).

## 8. What Changed From The Torch Build (PR #11) And The Old Stack

- **The probability model is a classifier on the children**, not a Dirichlet regression
  of the shares (PR #11) nor the old stack's grouped multinomial: the multinomial
  likelihood of the counts as categorical rows, with any scikit-learn classifier.
- **Calibration is inside the model** (`CalibratedClassifierCV`, cross-fitted on folds of
  buildings), a setting a tuner reaches, with three methods; not a post-hoc
  `TemperatureCalibrator` fitted by the caller, nor the old stack's gated calibration.
- **The total is `CountModel`**, the same class as Model 1, with any regressor; NB2 is
  statsmodels' through `NegativeBinomialRegressor`, given the raw exposure. No clipping,
  no floor on $\alpha$; a failed fit raises.
- **No torch** and no hand-written objective or optimizer (`optimization.py` is gone);
  penalties and solvers are the estimators' own; the exposure is an argument, built once
  with `ExposureTransformer`; hyperparameters live in the constructors and are tuned from
  outside; means only (no draws, intervals, bootstrap refits, state bundles or metadata).

## 9. Evidence

The smoke run (ten simulated populations, grouped 80/20 by neighborhood, class defaults,
untuned; Model 1 = `IndependentCohortModels` of LightGBM `CountModel`s on the §8.1
features, Model 2 on `total_base` / `cohort_probability_base`; held-out Poisson deviance
and `COHORT_LOG_LOSS`, mean ± SD). The full tables, the paired differences and the recipe:
[plan, sub-task 6](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md).

| Model | `n_kindergarten` | `n_elementary` | `n_highschool` | total | cohort log loss |
|---|---|---|---|---|---|
| baseline (constant rate) | 2.739 ± 0.680 | 2.532 ± 0.492 | 2.936 ± 0.909 | 5.277 ± 1.722 | 1.089 ± 0.007 |
| Model 1 | 2.628 ± 0.877 | 2.128 ± 0.499 | 2.572 ± 0.710 | 4.215 ± 1.450 | 1.093 ± 0.010 |
| Poisson total × `LogisticRegression()` | 2.485 ± 0.531 | 2.159 ± 0.453 | 2.324 ± 0.618 | 4.318 ± 1.272 | 1.082 ± 0.010 |
| … × LR, temperature | 2.428 ± 0.543 | 2.160 ± 0.446 | 2.344 ± 0.629 | 4.318 ± 1.272 | 1.082 ± 0.008 |
| … × `LGBMClassifier()` / temperature | 2.825 / 2.512 | 2.344 / 2.191 | 2.660 / 2.612 | 4.318 ± 1.272 | 1.101 / 1.090 |
| … × `HistGradientBoostingClassifier()` / temperature | 3.013 / 2.547 | 2.372 / 2.186 | 2.640 / 2.594 | 4.318 ± 1.272 | 1.106 / 1.091 |
| … × `RandomForestClassifier()` / temperature | 2.634 / 2.495 | 2.276 / 2.205 | 2.524 / 2.573 | 4.318 ± 1.272 | 1.093 / 1.089 |
| NB2 total (`NegativeBinomialRegressor`) | | | | 4.347 ± 1.306 | |
| LightGBM Poisson total | | | | 4.512 ± 1.603 | |

**Reading.** Model 1 and the baseline reproduce PR #11's B7 to the third decimal, and
NB2's $\alpha$ (0.103 ± 0.014) and total (4.347 vs 4.348) are the torch build's. On the
total the GLM totals are level with Model 1 ($t$ +0.5 Poisson, +0.4 NB2); the LightGBM
total is worse ($t$ +2.0: at the defaults on `total_base`'s nine scaled columns it is
weaker than Model 1's three LightGBMs on `tree`, which keep `n_apartments` as a feature).
sklearn's `PoissonRegressor()` default `alpha=1` is penalized, so B7's seed-1 overfit
(6.90) is gone (3.60). On the composition `LogisticRegression()` wins by a small amount:
0.010 below Model 1 ($t$ −5.4, 9 of 10) and 0.007 below the marginal shares ($t$ −3.6);
1.082 against the torch Dirichlet's 1.083 calibrated; still under 1 % per child. The tree
classifiers at their defaults are overconfident (`beta_` 0.39–0.55); raw, LightGBM and
HGB are worse than the marginal shares ($t$ +3.4 and +4.6; RF within noise), and
temperature repairs most of it (§3.4). Nothing called for a code change; the tunables
are the estimators' own (the trees' depth and learning rate, LR's `C`, the GLM
penalties).
