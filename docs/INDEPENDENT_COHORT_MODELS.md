# Independent Cohort Models (Model 1)

> The rebuilt Model A / Model 1 in `age_group_prediction.modeling`: `CountModel` (one
> count column, any regressor, the exposure as an offset) and `IndependentCohortModels`
> (one `CountModel` per cohort). Built and tested; nothing calls them yet. The original
> `models/direct_cohort.py`, which `experiment/` and `tracking/` still run, is described in
> [DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md) §1–§10 until the old stack is deleted
> ([roadmap](MODEL_REIMPLEMENTATION_PLAN.md)). `CountModel` is also Model 2's total model
> ([Total times probability model §2](TOTAL_TIMES_PROBABILITY_MODEL.md#2-the-total-countmodel)),
> so it is explained here only. Decisions and evidence: PR #12
> ([plan](DIRECT_COHORT_GENERALIZATION_PLAN.md)), PR #10 ([plan](MULTI_COHORT_MODELS_PLAN.md)),
> PR #13 ([plan](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md), P2–P4, P15, §6).

Each component below: what it is, the formula, the derivation, why, the code, the
measured check.

## 1. The Model

Building $b$ has $E_b > 0$ apartments (the exposure), features $x_b$, and cohort counts
$C_{b,k}$, $k = 1, \dots, K$. Model 1 fits each cohort **independently**, as a regression
of its count whose mean is proportional to the exposure:

$$
\mu_{b,k} = E_b\, f_k(x_b),
\qquad
\widehat C_{b,k} = E_b\, \hat f_k(x_b),
$$

with $f_k$ the cohort's own regressor (a Poisson or Gaussian loss, or NB2) on its own
features. Nothing ties the cohorts together: each is a `CountModel`, and
`IndependentCohortModels` holds one per column of `y`.

## 2. `CountModel`

One regressor for one count column, a cohort's or a total. `estimator` is required and
carries the loss and the hyperparameters (`LGBMRegressor(objective="poisson")`,
`HistGradientBoostingRegressor(loss="poisson")`, `PoissonRegressor()`,
`NegativeBinomialRegressor()`, `LGBMRegressor(objective="regression")`, …); a tuner
reaches them by nested names (`estimator__n_estimators`). `X` is the raw table when
`feature_transformer` is given, otherwise the finished design matrix: `fit` fits a copy
of the transformer on the rows it is fitted on and `predict` transforms new rows with
that copy, so test rows get the training statistics
([Feature transformations §8.1](FEATURE_TRANSFORMATIONS.md#81-model-a--countmodel-lightgbm)).
With `use_exposure=True` the raw exposure is passed to `fit` and `predict` and enters the
estimator in one of two ways, §2.1 and §2.2, chosen by the estimator's `fit` signature.

### 2.1 The exposure as a weighted rate

**What.** For an estimator whose `fit` takes `sample_weight`, `CountModel` fits the
observed rate $r_b = y_b / E_b$ with weight $E_b$, and `predict` multiplies the fitted
rate back by the exposure. The two losses it serves are different models, derived
separately below; the rule is the same for both.

**(a) Poisson loss.** The model learns the rate per apartment $\lambda_b = f(x_b) =
e^{F(x_b)}$, $F$ the raw score (the trees' sum in LightGBM, $\beta_0 + x\beta$ in a GLM),
and the count is $y_b \sim \text{Poisson}(\mu_b)$ with $\mu_b = E_b\lambda_b$.

- *(A) The offset model.* $\log\mu_b = \log E_b + F(x_b)$. Dropping $\log y_b!$, the
  negative log-likelihood is $\mathcal{L}_{\text{off}}(F) = \sum_b [E_b e^{F_b} - y_b F_b]
  - \sum_b y_b \log E_b$.
- *(B) The weighted rate.* The estimator's Poisson loss of a non-negative real target
  $r$ with mean $\lambda$ is $\lambda - r\log\lambda$ (scikit-learn and LightGBM define
  it for any $r \ge 0$, so a fractional rate is a valid target). With weight $w_b = E_b$:
  $\mathcal{L}_{\text{rate}}(F) = \sum_b E_b[e^{F_b} - r_b F_b] = \sum_b [E_b e^{F_b} -
  y_b F_b]$.
- *(C) Equivalence.* $\mathcal{L}_{\text{off}} = \mathcal{L}_{\text{rate}} - \sum_b y_b
  \log E_b$, a constant in $F$: the same minimizer, and the same per-row gradient and
  Hessian, $\partial\mathcal{L}/\partial F_b = E_b e^{F_b} - y_b = \mu_b - y_b$,
  $\partial^2\mathcal{L}/\partial F_b^2 = \mu_b$, which is all a boosting step uses to grow
  a tree. The starting point agrees too: with weights LightGBM's `boost_from_average`
  starts at $\log(\sum_b w_b r_b / \sum_b w_b) = \log(\sum_b y_b / \sum_b E_b)$, the
  intercept the old code put into `init_score`. Hence $\hat\mu_b = E_b \cdot
  \texttt{estimator\_.predict}(x_b)$: the weighted rate **is** the Poisson offset model.

**Check (Poisson).** LightGBM (2,000 rows, 50 trees): the weighted-rate prediction and
LightGBM's own `init_score` offset agree to $1.5 \times 10^{-8}$ relative
(`test_the_weighted_rate_equals_lightgbms_offset`); statsmodels' Poisson GLM with
`offset=log E` and with the weighted rate: coefficients within $2 \times 10^{-13}$
(2026-10-07).

**(b) Gaussian loss.** Here $f$ enters the mean directly (identity link), $f_b$ short for
$f(x_b)$.

- *(D1) The model.* The count is Gaussian with a mean proportional to the exposure and a
  variance that grows with it, as a count's does: $y_b \sim \mathcal{N}(\mu_b, \sigma^2
  E_b)$, $\mu_b = E_b f_b$, with negative log-likelihood $\mathcal{L}_{\text{Gauss}}(f,
  \sigma^2) = \sum_b \big[(y_b - E_b f_b)^2 / (2\sigma^2 E_b) + \tfrac12\log(2\pi\sigma^2
  E_b)\big]$.
- *(D2) The weighted rate.* The estimator's squared-error loss $\tfrac12(r - f)^2$ of the
  rate with weight $w_b = E_b$: $\mathcal{L}_{\text{rate}}(f) = \tfrac12\sum_b E_b (r_b -
  f_b)^2 = \tfrac12\sum_b (y_b - E_b f_b)^2 / E_b$.
- *(D3) Equivalence.* $\mathcal{L}_{\text{Gauss}}(f, \sigma^2) = \mathcal{L}_{\text{rate}}(f)
  / \sigma^2 + \tfrac12\sum_b\log(2\pi\sigma^2 E_b)$, and the last term is free of $f$: for
  every $\sigma^2$ the same minimizer in $f$, so the prediction never needs $\sigma^2$.
  Per row, $\partial\mathcal{L}_{\text{rate}}/\partial f_b = E_b f_b - y_b = \mu_b - y_b$ (the
  Poisson form) and $\partial^2\mathcal{L}_{\text{rate}}/\partial f_b^2 = E_b$ (the exposure,
  where Poisson has $\mu_b$). A constant $f$ minimizes it at $f = \sum_b y_b / \sum_b E_b$,
  the weighted mean rate a booster starts from. Again $\hat\mu_b = E_b \cdot
  \texttt{estimator\_.predict}(x_b)$.
- *(D4) Why not the additive offset, and why the weight.* With an identity link the
  offset of (A) gives the mean $f(x_b) + \log E_b$, not proportional to the apartments;
  that is why the old model rejected `use_exposure` with a Gaussian loss. The weighted
  rate scales the mean instead, so `use_exposure=True` is meaningful for a Gaussian loss
  too. Without the weight (a plain regression of $r_b$) the mean would be the same but
  every building would count equally, a $\operatorname{Var}(y_b) \propto E_b^2$
  assumption. The one Gaussian caveat: $f(x_b)$ can be negative, as it already could
  without an exposure.

**Check (Gaussian).** The weighted-rate fit equals the weighted least-squares fit of $y$
on $[E, Ex]$ with weights $1/E$ to $10^{-16}$, and the full maximum-likelihood fit of
(D1), $\sigma^2$ included, to $10^{-8}$
(`test_the_gaussian_weighted_rate_is_least_squares_of_the_count`).

**(c) Why not a residual.** Fitting $y_b - \mu^{(0)}_b$ (the count minus an initial
prediction) is only meaningful for a squared-error loss, where the residual is again a
Gaussian target; for a Poisson loss it is not a count and can be negative, so a residual
path would fit a different model than the LightGBM one. The model uses the rate for every
estimator instead.

**Why.** One rule serves every estimator with `sample_weight` (LightGBM, HGB, sklearn's
GLMs, linear regression) with no `init_score` or custom objective: exact for the Poisson
loss, and for the Gaussian loss the intended mean structure (expected count proportional
to apartments, $f$ the average children per apartment).

**Code.** `CountModel.fit`: `estimator.fit(X, y / exposure, sample_weight=exposure)`;
`predict`: `estimator_.predict(X) * exposure`.

**The penalty under the rate form.** A penalized scikit-learn GLM minimizes
$\tfrac{1}{\sum_b w_b}\sum_b w_b\,\text{loss}_b + \tfrac{\alpha}{2}\lVert\beta\rVert^2$:
it **normalizes** `sample_weight` (scaling every weight by a constant leaves the fit
unchanged, measured $1.4 \times 10^{-17}$). Multiplying through by $\sum_b E_b$ gives
$\sum_b \text{loss}_b + \tfrac{\alpha\sum_b E_b}{2}\lVert\beta\rVert^2$, while the offset
model as scikit-learn fits it without weights is $\tfrac1n\sum_b \text{loss}_b +
\tfrac{\alpha'}{2}\lVert\beta\rVert^2$, i.e. $\sum_b \text{loss}_b +
\tfrac{n\alpha'}{2}\lVert\beta\rVert^2$. They coincide at $\alpha' = \alpha \sum_b E_b / n
= \alpha\bar E$: under the rate form `PoissonRegressor(alpha)` acts as `alpha` $\times
\bar E$ in the offset model (measured 2026-10-05: with `alpha=1` the rate fit matches the
offset model's at `alpha` $= \bar E = 47.6$ to $10^{-4}$). LightGBM and HGB use the weights
as given, so their penalties are unchanged; no single rescaling of the weights fixes both
(dividing by $\bar E$ would change LightGBM's Hessians and so its `reg_lambda` and
`min_child_weight`). Penalties are therefore the estimators' own, and each tuned in its
own units.

### 2.2 The exposure passed to the estimator

**What.** An estimator whose `fit` has an `exposure` parameter (the `ExposureRegressor`
protocol: `fit(X, y, exposure=None)`, `predict(X, exposure=None)`) is given the raw
exposure itself, at `fit` and at `predict`, and forms the offset inside.

**Derivation (why the rate is not NB2).** The identity in §2.1 (C) rests on the Poisson
loss being linear in $y$ and homogeneous in $(y, \mu)$: $E[\lambda - r\log\lambda] =
[E\lambda - y\log(E\lambda)] + y\log E$. The NB2 log-likelihood of a count $y$ with mean
$\mu$ and dispersion $\alpha$,

$$
\ell(y;\mu,\alpha) = \log\frac{\Gamma(y + 1/\alpha)}{\Gamma(1/\alpha)\,\Gamma(y+1)}
+ \frac{1}{\alpha}\log\frac{1}{1 + \alpha\mu}
+ y\log\frac{\alpha\mu}{1 + \alpha\mu},
$$

is not: under the offset model $\mu_b = E_b\lambda_b$ its dispersion term is
$\tfrac{1}{\alpha}\log\tfrac{1}{1 + \alpha E_b\lambda_b}$, while the weighted rate
evaluates $\ell$ at $r_b$ with mean $\lambda_b$ and multiplies by $E_b$, giving
$\tfrac{E_b}{\alpha}\log\tfrac{1}{1 + \alpha\lambda_b}$. The variance $\mu(1 + \alpha\mu)$
has a term in $E_b^2$, so the two are different functions of $\lambda_b$: the rate form
fits a different NB2 model, and statsmodels accepts the non-integer rate silently.

**statsmodels' `exposure` is the offset.** statsmodels adds $\log(\texttt{exposure})$ to
the linear predictor with coefficient 1, so `exposure=E` and `offset=log(E)` are the
same model (measured: identical parameters, 0.0). The argument is the raw exposure, as
every model here takes it, and **the exposure only, no `offset`**: no model here has a
second fixed log-scale term, and the two would add up silently if both were passed.

**Why a signature test.** The case is read from the estimator's own `fit` with
scikit-learn's `has_fit_parameter` (its idiom for `sample_weight`, as in
`BaggingRegressor`): it names the capability, not one class, so any future estimator
taking the exposure is treated the same. Order: `exposure`, then `sample_weight`, else
`TypeError` under `use_exposure=True`. A wrapper whose `fit` takes `**fit_params`
(`Pipeline`, `TransformedTargetRegressor`, `GridSearchCV`) is refused even if it would
forward `sample_weight` (measured: `TransformedTargetRegressor` used to fit on the rate):
pass the regressor itself; the feature transformer belongs to `CountModel`. Without
`use_exposure` every estimator is fitted plainly and a passed exposure is ignored.
`predict` decides by the fitted copy, `estimator_`, not the current setting.

**Code.** `count_model.py`: `ExposureRegressor`, `_takes_exposure` /
`_takes_sample_weight` (`TypeGuard`s over `has_fit_parameter`), the three named cases in
`fit` and `predict`. **Check.** `has_fit_parameter(…, "exposure")` is `False` for
`PoissonRegressor`, `LinearRegression`, HGB and `LGBMRegressor`, `True` for
`NegativeBinomialRegressor`; `GLM(NegativeBinomial(alpha))` with `offset=log E` vs the
weighted rate: 0.021 (2026-10-07; 0.012 on another draw, 2026-10-08).

### 2.3 `NegativeBinomialRegressor`

**What.** NB2 as a scikit-learn regressor (`BaseEstimator`, `RegressorMixin`) around
statsmodels' `discrete_model.NegativeBinomial(loglike_method="nb2", exposure=…)`, an
`ExposureRegressor`:

$$
\log\mu = \log E + b + X\beta,
\qquad
\operatorname{Var}(Y) = \mu(1 + \alpha\mu),
$$

with $b$, $\beta$ and $\alpha$ fitted jointly by maximum likelihood ($\alpha$ is free,
unlike a GLM with a fixed $\alpha$). `predict(X, exposure=E)` is $E\cdot\exp(b + X\beta)$
from the stored `intercept_` and `coef_` (what statsmodels' `results.predict(exog,
exposure=E)` computes; $1.1 \times 10^{-14}$); `dispersion_` is $\alpha$. One setting,
`max_iter=500`; unpenalized.

**The fit, and why each choice.** BFGS (`method="bfgs"`), the preliminary Poisson fit too
(`optim_kwds_prelim`): statsmodels' default Newton preliminary fit raises `LinAlgError` on
an all-zero or constant column (a dummy absent from a fold), BFGS fits it.
`add_constant(X, has_constant="add")`: the default `"skip"` adds no intercept beside a
constant column, shifting the parameters by one, so `params[0]` would be a coefficient.
`skip_hessian=True`: the Hessian only gives standard errors, unused, and cannot be
inverted with such a column (its `HessianInversionWarning` appeared on converged fits).
statsmodels' `ConvergenceWarning` is silenced inside `fit` and non-convergence raises our
`RuntimeError`: one signal with the same information. `y` is not checked to be integer
(`validate_data` checks it is numeric and finite): counts come from the data
preparation, `PoissonRegressor` does not check either, and statsmodels fits
a non-integer `y` silently ($\alpha$ 0.038 instead of 0.096 on `y + 0.5`). **Code:**
`negative_binomial.py`.

**Why statsmodels' discrete NB, and not:** `GLM(family=NegativeBinomial(alpha))` ($\alpha$
must be fixed; its `fit_regularized(alpha, L1_wt=0)` is an L2 variant for later if tuning
needs one); PR #11's torch objective (no torch in `modeling/`); `glum` (not installed,
$\theta$ fixed); a LightGBM / XGBoost custom objective (no NB objective; it would re-create
the torch code); scikit-learn (`TweedieRegressor` is not NB).

**Check.** On the simulator's totals (245 rows, 7 features): converged in 6 ms, $\alpha$
0.1002 for a true 0.1, BFGS vs Newton $8.5 \times 10^{-8}$; recovery on 5,000 NB2 draws:
$b$ −2.011 (true −2), $\beta$ (0.302, −0.193, 0.099), $\alpha$ 0.1014 (0.1);
`CountModel(NegativeBinomialRegressor(), use_exposure=True)` equals statsmodels'
`exposure=` fit to $10^{-5}$
(`test_nb2_through_the_model_equals_statsmodels_with_the_exposure`); over the ten
smoke-run populations $\alpha$ 0.103 ± 0.014, the torch build's figure
([Total times probability model §9](TOTAL_TIMES_PROBABILITY_MODEL.md#9-evidence)).

## 3. `IndependentCohortModels`

**What.** `IndependentCohortModels(cohort_models)` maps each cohort (a column of `y`) to
its model, usually a `CountModel` with its own `feature_transformer`, so each cohort
keeps its own features and hyperparameters. `fit` fits a **copy** of each on the raw
table and its column of `y` (stored in `cohort_models_`, set only once every cohort
succeeded); `predict` returns a DataFrame with one column per cohort, in `y`'s column
order at fit whatever the mapping's order, indexed like `X`.

**Why these rules.**
- *The same exposure goes to every cohort*; a model with `use_exposure=False` ignores it,
  one fitted with the offset raises without it. One exposure serves every model, a tuner
  compares with and without the offset on one fixed exposure, and a forgotten exposure
  cannot drop the offset silently.
- *Rows are paired by position*, not by index, as in scikit-learn: the splitter returns
  positions and every array is taken by them. Predictions are placed by position too: a
  prediction with its own index would be realigned to `X`'s, into NaN, silently.
- *The keys of `cohort_models` equal `y`'s columns, each once*, or `fit` raises
  (`Counter`, not sets: a duplicated column would reach its model as a DataFrame).
- *The targets come from `y` only.* With `remainder="drop"` (the §8 transformers) `X` may
  keep the target columns; `predict` needs none.
- *Each cohort is tuned on its own.* A nested name such as
  `cohort_models__n_kindergarten__estimator__learning_rate` raises `AttributeError`: tune
  each `CountModel` (its nested names work), then assemble the mapping or replace it with
  `set_params(cohort_models=...)`. The cohorts are independent, so no study tunes them
  together.
- *Set before `fit`.* The models and their estimators and transformers are cloned in
  `fit`; `predict` follows the fitted copies, so a refit equals a fresh fit (the contract
  test). A plain `dict` suffices for the mapping.

**Code.** `independent_cohorts.py`. **Check.** The A4 smoke run through
`IndependentCohortModels` gives per-cohort numbers identical to the three `CountModel`s
fitted alone (§8), and `test_each_column_equals_that_cohorts_model_fitted_alone` pins it.

## 4. API

Import from `age_group_prediction.modeling`; the package root exports the **original**
classes of the old stack.

| Class | Settings (defaults) | `fit` takes | Fitted state | `predict` returns |
|---|---|---|---|---|
| `CountModel` | `estimator` (required: a `Regressor`, `fit(X, y, sample_weight=None)` / `predict(X)`, or an `ExposureRegressor`, `fit(X, y, exposure=None)` / `predict(X, exposure=None)`); `use_exposure=False`; `feature_transformer=None` | the raw table with `feature_transformer`, otherwise the design matrix; one count column; `exposure=` (the raw $E$), required with `use_exposure`, ignored without | `estimator_`, `use_exposure_`, `feature_transformer_` | the mean count, a 1-D array; it follows how the model was fitted (an exposure is required if it was fitted with one) |
| `NegativeBinomialRegressor` | `max_iter=500` | `X`, the counts `y`, `exposure=None` (raw) | `intercept_`, `coef_`, `dispersion_` ($\alpha$), `n_features_in_`, `feature_names_in_` | $E\cdot\exp(b + X\beta)$, or $\exp(b + X\beta)$ without an exposure |
| `IndependentCohortModels` | `cohort_models` (a mapping cohort → `BaseAgeGroupModel`) | the raw table, a DataFrame of cohort counts, `exposure=` for every cohort | `cohort_models_` | a DataFrame, one column per cohort in `y`'s order, indexed like `X` |

Every model has `evaluate(y_true, y_pred, metric)` with a `Metric`
(`POISSON_DEVIANCE`, `RMSE`, `MAE`, or a custom one). Build LightGBM with `n_jobs=1`
while the old stack's tests still import torch: more OpenMP threads crash alongside it
on macOS. LightGBM ignores `subsample` unless `subsample_freq >= 1` is set too.

## 5. Data Flow

1. **Row-wise preprocessing, on the full table, before the split.** `ShareTransformer`
   turns the room counts into shares; `ExposureTransformer` returns the exposure as
   floats and raises for a zero, negative, infinite or NaN value (LightGBM would fit it
   silently; HGB raises). Neither learns anything, so nothing leaks from the test rows,
   and a bad test row fails before any model is fitted.
2. **Split by neighborhood** with `Splitter.train_test_indices`: row positions, applied
   to the table, the targets and the exposure alike with `take_rows`.
3. **Fit on the training rows.** Each model fits its feature transformer inside its own
   `fit`, on those rows only.

```python
from lightgbm import LGBMRegressor
from age_group_prediction.modeling import CountModel, IndependentCohortModels
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import POISSON_DEVIANCE
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

# 3. One model per cohort, each with its own transformer; tree is Feature transformations §8.1
def lightgbm():
    return LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)

model = IndependentCohortModels({
    "n_kindergarten": CountModel(estimator=lightgbm(), use_exposure=True, feature_transformer=tree),
    "n_elementary": CountModel(estimator=lightgbm(), feature_transformer=tree),  # ignores the exposure
    "n_highschool": CountModel(estimator=lightgbm(), use_exposure=True, feature_transformer=tree),
}).fit(X_train, Y_train, exposure=exposure_train)
predictions = model.predict(X_test, exposure=exposure_test)  # a DataFrame
scores = {
    cohort: model.evaluate(Y_test[cohort], predictions[cohort], POISSON_DEVIANCE)
    for cohort in COHORTS
}
```

Run as written on the simulator's seed-0 population (2026-10-08, under `-W error`):
196 training and 49 test buildings; `scores` 1.903 (kindergarten), 2.011 (elementary),
2.404 (high school).

**Correct inputs.** `y` is the raw count, never the rate or its log (the model divides by
the exposure itself); `exposure` is the raw $E_b$, never $\log E_b$ (which would be
divided into `y` as if it were apartments); the output is an expected **count**. If the
exposure is not a column of `X`, doubling it doubles the prediction exactly; if `X`
includes `n_apartments`, $F$ can learn departures from proportionality.

## 6. Errors

The models check only what would otherwise pass silently; values are validated where the
data is prepared.

| Raises | When |
|---|---|
| `ValueError` "pass `exposure`" / "inconsistent numbers of samples" / "one-dimensional" | a model fitted or used with the offset but no exposure, or one of another length, or a column `(n, 1)` that would broadcast into an $(n, n)$ prediction (`BaseAgeGroupModel._check_exposure`) |
| `TypeError` "takes neither `exposure` nor `sample_weight`" | `use_exposure=True` with an estimator (or a `**fit_params` wrapper) whose `fit` takes neither |
| `RuntimeError` "NB2 did not converge" | `NegativeBinomialRegressor` stopped at `max_iter`; also the outcome of an exposure of 0, negative or NaN |
| `ValueError` "exposure has shape" | an exposure that is not one value per row at `NegativeBinomialRegressor.predict` |
| `ValueError` "cohort_models has … but y's columns are" | the mapping's keys differ from `y`'s columns |
| `NotFittedError` | `predict` before `fit` |

Left to the estimator: an unknown objective, an all-zero `y` under a Poisson loss
(LightGBM raises; not every estimator does), NaN in `X`, and reordered columns without a
feature transformer (LightGBM accepts them silently; the transformer selects by name;
[plan §10](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md)).

## 7. What Changed From The Old Stack (`models/direct_cohort.py`)

- One instance per cohort with the hyperparameters fixed on the estimator and tuned from
  outside (`estimator__…`), instead of one instance for three cohorts tuning with Optuna
  inside `fit`.
- Any regressor as `estimator`, instead of LightGBM only with `poisson` / `nb2` (a custom
  gradient) / `normal`; NB2 is statsmodels' `NegativeBinomialRegressor`.
- An optional `feature_transformer` fitted inside `fit`, instead of a `FeatureSpec` and
  fitted preprocessing owned by the model; an optional exposure (§2.1–§2.2), where the
  old model had none (PR #6's first rebuild put it into LightGBM's `init_score`; PR #12
  replaced that with the weighted rate, the same trees).
- Means only: no bootstrap draws, intervals, pointwise log probabilities,
  `PredictionResult`, state bundles or metadata; `get_params()` and the fitted attributes.

## 8. Evidence

**Does the exposure help?** Ten simulated populations, split by neighborhood, untuned
defaults, `n_apartments` kept as a feature: the exposure lowered held-out Poisson
deviance by about 4 % for kindergarten and high school and not at all for elementary;
through `IndependentCohortModels` the summed prediction's deviance against
`n_children_total` is 4 % lower, in 8 of 10 populations
([multi-cohort plan, A4](MULTI_COHORT_MODELS_PLAN.md#a4-smoke-run-and-docs)). Weak
evidence, untuned: re-check once the models are tuned. Model 1 at the class defaults is
the reference row of Model 2's smoke run
([Total times probability model §9](TOTAL_TIMES_PROBABILITY_MODEL.md#9-evidence)):
deviance 2.628 / 2.128 / 2.572 per cohort, 4.215 ± 1.450 for the total, cohort log loss
1.093 ± 0.010.
