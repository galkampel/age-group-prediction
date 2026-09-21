# Feature Engineering

How building-level features are declared, fitted, transformed, persisted, and
consumed by each model. Everything here is derived from
`src/age_group_prediction/fitted_features.py`,
`src/age_group_prediction/modeling_config.py`, and
`src/age_group_prediction/experiment/candidates.py`; the design rationale is in
[MODELING_REBUILD_PLAN.md](MODELING_REBUILD_PLAN.md) Section 5.

## 1. Two Lifecycles

| Stage | Owner | Learns from data? | When it runs |
|---|---|---|---|
| Modeling-table construction | `build_modeling_table` (`dataset_builder.py`) | No: deterministic selection, room-share derivation, validation | Once, before splitting |
| Feature transformation | `FittedFeatureTransformer` (`fitted_features.py`) | Yes: scaling and spline knots | Separately inside every fit partition (fold, tuning fold, full refit) |

Keeping them apart is what prevents leakage: nothing that depends on the data's
distribution is computed before the split, and every learned statistic comes
only from the rows a model is fitted on.

## 2. Inputs: The Modeling Table

`build_modeling_table(source_df)` produces exactly these columns, in this
order, and validates them against `ModelingSchema`:

| Role | Columns |
|---|---|
| Identifiers (never features) | `building_id`, `neighborhood_id` |
| Raw numeric features | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, `n_apartments` |
| Derived room shares | `3_rooms_share`, `4_rooms_share`, `5_rooms_share` |
| Categorical feature | `school_status` with levels `none` (reference), `existing`, `planned` |
| Targets (never features) | `n_kindergarten`, `n_elementary`, `n_highschool`, `n_children_total` |

Room shares are `k_rooms / n_apartments`. The 6-room share is the omitted
reference, so the three shares plus the reference sum to one and no collinear
column enters a design matrix. Raw room counts (`3_rooms` … `6_rooms`) are not
carried into the table, and a `FeatureSpec` may not combine room counts with
room shares.

The schema also forbids identifiers, targets, and the simulator's latent
effect names (`u_b`, `w_j`, `v_b`) as features.

## 3. Declaring Features: `FeatureSpec`

A `FeatureSpec` (frozen, JSON-serializable) declares one model component's
representation. It holds no data.

| Field | Values (default) | Meaning |
|---|---|---|
| `component` | `tree`, `total_count`, `composition` | Which model part this spec feeds; drives validation |
| `numeric_features` | tuple of schema numeric columns | Base numeric main effects |
| `categorical_features` | tuple of schema categorical columns | One-hot encoded against the schema reference |
| `ses_form` | `linear` (default), `quadratic`, `spline` | SES representation |
| `daycare_form` | `linear` (default), `log1p` | Daycare-count representation |
| `interactions` | subset of `ses_x_household_size`, `daycare_x_median_age`, `room_share_x_household_size`, `room_share_x_median_age` (`()`) | Explicit interaction blocks |
| `scale_numeric` | `False` | Standardize numeric features with fit-partition mean and standard deviation |
| `exposure_column` | `None` or the schema exposure (`n_apartments`) | Log-exposure offset for count models |
| `unknown_category_policy` | `error` (default), `treat_as_reference` | Behavior for categorical values outside the schema |
| `spline_n_knots` | `4` | Knots for the cubic SES spline (minimum 3) |
| `ses_column`, `household_size_column`, `median_age_column`, `daycare_column` | schema column names | Semantic roles used by forms and interactions |
| `specification_version` | `"1"` | Recorded in metadata and bundles |

### Validation rules

Construction (`FeatureSpec.__post_init__`) refuses:

- unknown components, forms, policies, or interactions; duplicate columns;
  columns listed as both numeric and categorical;
- a spline with fewer than 3 knots;
- spline SES together with `ses_x_household_size` (the spline replaces the
  linear SES column the interaction would need);
- `room_share_x_household_size` on `composition`, and
  `room_share_x_median_age` on `total_count`;
- an exposure on any component other than `total_count`, or an exposure that
  is also listed as an ordinary numeric feature.

`validate_for_schema(schema)`, run before any fit, refuses:

- columns the schema does not model, and identifiers, targets, or forbidden
  latent columns;
- raw room counts mixed with room shares;
- an exposure column different from the schema's;
- an interaction or daycare form whose main effects are not selected (for
  example `daycare_x_median_age` without both `n_daycares_500m` and
  `median_age`). `ses` must always be selected.

### Named defaults

| Spec | `component` | Numeric features | Scaling | Exposure |
|---|---|---|---|---|
| `DEFAULT_TREE_FEATURE_SPEC` | `tree` | all 8 numeric columns, including `n_apartments` as an ordinary feature | No | None |
| `DEFAULT_TOTAL_FEATURE_SPEC` | `total_count` | the 7 non-exposure numeric columns | Yes | `n_apartments` (log offset) |
| `DEFAULT_PROBABILITY_FEATURE_SPEC` | `composition` | the 7 non-exposure numeric columns | Yes | None |

All three include `school_status`, use linear SES and daycare forms, and have
no interactions.

## 4. Fitting And Transforming: `FittedFeatureTransformer`

```python
from age_group_prediction import DEFAULT_TOTAL_FEATURE_SPEC, FittedFeatureTransformer

transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC)
fit_features = transformer.fit_transform(fit_df)      # learns state from fit_df only
eval_features = transformer.transform(eval_df)        # reuses that state
log_exposure = transformer.get_log_exposure(eval_df)  # separate offset, or None
names = transformer.get_feature_names_out()
```

`transform` before `fit` raises. `fit_transform` returns the frame built during
`fit` rather than transforming twice.

### Input checks (every `fit` and `transform`)

Selected numeric, categorical, and exposure columns must be present; numeric
values finite; categorical values non-missing; unknown categories refused under
the `error` policy; exposure finite and strictly positive.

### Transformation order

For each partition, `_transform_fitted` builds the output in this order:

1. **Numeric main effects.** With `scale_numeric`, each column becomes
   `(x - mean_fit) / sd_fit` using population standard deviation from the fit
   partition; a constant column keeps scale 1 instead of dividing by zero.
2. **SES form.**
   - `linear`: `ses` as is.
   - `quadratic`: adds `ses_squared`, the square of the (scaled) `ses` column.
   - `spline`: replaces `ses` with a cubic B-spline basis fitted on raw SES from
     the fit partition. With the default 4 knots this is five columns,
     `ses_spline_0` … `ses_spline_4`.
3. **Daycare form.** `log1p` replaces `n_daycares_500m` with
   `n_daycares_500m_log1p`, computed from raw non-negative counts.
4. **Interactions.** Each term multiplies the variables exactly as the matrix
   carries them as main effects (scaled, or `log1p` for daycare), so hierarchy
   holds in the output:

   | Interaction | Output columns |
   |---|---|
   | `ses_x_household_size` | `ses_x_avg_household_size` |
   | `daycare_x_median_age` | `n_daycares_500m_x_median_age` |
   | `room_share_x_household_size` (total only) | `3_rooms_share_x_avg_household_size`, `4_…`, `5_…` |
   | `room_share_x_median_age` (probability only) | `3_rooms_share_x_median_age`, `4_…`, `5_…` |

5. **Categoricals.** One-hot encoding with the schema's reference level dropped:
   `school_status_existing`, `school_status_planned`.
6. **Final check.** Every output value must be finite, and `transform` must
   reproduce the exact column names and order recorded at fit.

Default outputs (verified on simulator data):

| Spec | Output columns |
|---|---|
| Tree | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, `n_apartments`, `3_rooms_share`, `4_rooms_share`, `5_rooms_share`, `school_status_existing`, `school_status_planned` |
| Total and probability | the same without `n_apartments` (the total spec supplies `log_n_apartments` separately) |

### Categorical encoding and unknown levels

Category levels come from the schema, not from the fit data. A level absent
from a fold therefore still gets its column, and fitted state never depends on
which levels a fold happened to contain. Under `unknown_category_policy="error"`
an out-of-schema value is refused; under `"treat_as_reference"` it is encoded as
all zeros, which is indistinguishable from the reference level `none`. Missing
values are always refused.

### Exposure offset

`get_log_exposure(df)` returns `log(n_apartments)` named `log_n_apartments`
for the total-count spec and `None` otherwise. It is never a column of the
feature matrix: count models add it as a fixed offset, so expected children
scale with building size. It uses `log`, not `log1p`, because exposure is
required to be positive.

## 5. Fitted State, Metadata, And Persistence

| Method | Returns |
|---|---|
| `get_metadata()` | JSON-safe record: `feature_spec`, `feature_names`, `fit_row_count`, `scale_numeric`, `numeric_means`, `numeric_scales`, `spline_knots`, `category_references` |
| `to_state()` | Everything `transform` needs: spec, feature names, fit row count, means and scales, and spline base knots (no rows, no categories) |
| `FittedFeatureTransformer.from_state(state)` | An equivalent fitted transformer rebuilt without refitting; refuses state that disagrees with its spec about scaling or the spline |

Every model's state bundle embeds its transformer state, so a reloaded model
transforms new rows identically without the training data. The bundle still
contains fitted summary statistics (means, scales, knots); store it with the
same care as the data.

## 6. How Each Model Uses Features

| Model | Specs | Behavior |
|---|---|---|
| `DirectCohortModel` | One `tree` spec | Refuses non-tree specs and any exposure. Unscaled features; `n_apartments` is an ordinary predictor. A fresh transformer is fitted on each tuning fold and again on the full training data. |
| `IndependentTotalProbabilityModel` | `total_count` spec (fit argument, must have exposure) and `probability_feature_spec` (constructor) | Two independent transformers. Each component's regularization is tuned on training-only folds with transformers fitted per fold. The probability transformer is fitted on **all** training buildings; zero-total buildings then drop out of the grouped multinomial likelihood because their weights are zero, so the target never decides which rows set preprocessing state. |
| `BayesianConditionalModel` | `total_count` spec with exposure, plus `probability_feature_spec` | The total stage uses the base transformer and log exposure; composition uses its own transformer fitted on the same training rows. In cross-validation its specs must equal the selected independent model's frozen specs, or selection is refused. |

The base lifecycle (`models/base.py`) validates the spec against the schema,
fits the transformer on the training partition, passes the features and
log exposure to the model, and reuses the same transformer at prediction time.

## 7. Candidate Feature Specifications

`enumerate_feature_specs(component, *, include_daycare_saturation=False,
interactions=None)` (in `experiment/candidates.py`) returns an ordered, named
set of predeclared variants of a component's default spec. It never fits or
ranks anything.

| Component | Candidates, in order |
|---|---|
| `tree` | `tree__ses_linear`, `tree__ses_quadratic`, `tree__ses_spline`, and optionally `tree__daycare_log1p` |
| `total_count` | `total_count__ses_linear`, `__ses_quadratic`, `__ses_spline`, `__ses_x_household_size`, `__daycare_x_median_age`, `__room_share_x_household_size`, and optionally `__daycare_log1p` |
| `composition` | `composition__ses_linear`, `__ses_quadratic`, `__ses_spline`, `__ses_x_household_size`, `__daycare_x_median_age`, `__room_share_x_median_age`, and optionally `__daycare_log1p` |

Each interaction candidate adds exactly one interaction to the linear-SES base,
so any improvement is attributable. The daycare saturation form is opt-in
because the plan admits it only when held-out residuals justify it.

Feature forms are **not** hyperparameters. Optuna tunes only numerical
parameters inside a model's `fit`: LightGBM capacity for Model A, and the L2
penalty and `C` for Model B. A different feature form is a different candidate
compared on identical folds. See
[Hyperparameter Tuning](MODELING_GUIDE.md#hyperparameter-tuning).

The canonical registry (`build_canonical_candidate_registry`) uses the
`*__ses_linear` candidate for each component, which equals the named defaults.
The canonical comparison therefore compared model families, not feature forms;
other candidates are available for future, predeclared comparisons on the same
folds.

## 8. Guarantees

Covered by `tests/unit/test_fitted_features.py`:

- transform requires fit; transforming validation data never changes fitted
  state;
- fitted preprocessing is row-order invariant and never reads targets;
- output names are stable and all values finite for every candidate;
- categorical reference and unknown policy are schema driven, including a
  custom schema and a level absent from the fit fold;
- log exposure is separate from the matrix and requires positive values;
- component-specific interactions, main-effect hierarchy, and room
  count/share redundancy are rejected at spec construction or validation;
- probability preprocessing fits before zero-total filtering;
- spec and fitted metadata are JSON serializable; unscaled specs report no
  scaling state; interactions use the transformed main effects.

## 9. Pitfalls

- **Do not fit on data that includes validation or holdout rows.** Pass only
  the fit partition to `fit`/`fit_transform`.
- **Do not hand-assemble the modeling table.** Room shares, column order, and
  validation come from `build_modeling_table`.
- **Do not add `n_apartments` as a feature to a count spec.** It is refused;
  exposure belongs in the offset.
- **Do not reuse a transformer across folds.** Create a new one per fit
  partition; the models already do.
- **`treat_as_reference` hides new categories.** Prefer the default `error`
  and extend the schema when a new level is legitimate.
- **Changing a spec changes provenance.** Feature specs are part of candidate
  descriptors and fingerprints; a different spec is a different candidate.
- **Keep Bayesian specs tied to Model B.** Selecting a different form for the
  Bayesian candidate is refused by design.

## 10. Related Documents

- [MODELING_GUIDE.md](MODELING_GUIDE.md) Section 5 for the workflow context.
- [MODULE_REFERENCE.md](MODULE_REFERENCE.md) for module ownership.
- [FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md) for the rationale
  behind each transformation and proposed interpretability changes.
- [BAYESIAN_CONDITIONAL_MODEL.md](BAYESIAN_CONDITIONAL_MODEL.md) for how the
  Bayesian stages use the two design matrices.
