# Splitting

How the `splitting` package produces a train/test split and a matching
cross-validator, which method to choose, and what each one guarantees.

> **Not yet wired.** `data_splitting.py` and its seven consumers are still the
> production path, documented in [Data and splitting](DATA_AND_SPLITTING.md) and
> [Cross-validation and selection](CROSS_VALIDATION_AND_SELECTION.md). This
> package is built and tested but nothing calls it yet; §7 lists what remains.

## 1. The three methods

A method fixes *both* halves of an evaluation — the train/test split and the
cross-validator — so a run cannot pair one method's split with another's folds.
They differ in one thing: what a model may know about a test row's group.

| method | train/test split | cross-validator | a test group is... |
|---|---|---|---|
| `random` (default) | `ShuffleSplit(n_splits=1)` | `KFold(shuffle=True)` | usually in training |
| `stratified_by_group` | `StratifiedHoldout` | `StratifiedFolds` | **always** in training |
| `grouped` | `GroupShuffleSplit(n_splits=1)` | `GroupKFold(shuffle=True)` | **never** in training |

`StratifiedHoldout` and `StratifiedFolds` are ours; nothing in scikit-learn
splits *within* every group. The other four are scikit-learn's own.

Each method answers a different question:

- `random` — how well do I predict a new building?
- `stratified_by_group` — …in a neighborhood I already know?
- `grouped` — …in a neighborhood I have never seen?

## 2. The two-step API

```python
from age_group_prediction.splitting import Splitter

splitter = Splitter("stratified_by_group")

X_train, X_test, y_train, y_test, groups_train, groups_test = (
    splitter.train_test_split(X, y, groups, test_size=0.2, random_state=42)
)
cross_validate(
    estimator, X_train, y_train,
    cv=splitter.cv(n_splits=5, random_state=42), groups=groups_train,
)
```

Two steps, not one `n_splits + 1` way split: `train_test_split` is drawn once
and then fixed, while `cv` re-deals folds on every tuning pass. `train_test_split`
splits every array it is given and returns two per array in scikit-learn's
order; the groups come back because `cv` needs `groups_train`.

No parameter has a default — sizing and seeding are decisions the call site
states. `Splitter` is a frozen dataclass, so `repr` records the method.

## 3. What `stratified_by_group` guarantees

Every held-out or validated row keeps at least one row from its own group in the
fit set. A group of `n` rows gives up `max(1, min(round(n * test_size), n - 1))`
to the test set, so the realized share runs above `test_size` for small groups
and below it for singletons.

Coverage is graded by group size. At `test_size=0.2` and `n_splits=5`:

| rows in the group | → test | → train | validated in |
|---|---|---|---|
| 1 | 0 | 1 | never |
| 2 | 1 | 1 | never |
| 3-5 | 1 | n−1 | n−1 of 5 folds |
| 6 | 1 | 5 | all 5 |
| 7-9 | 1 | n−1 | all 5, one fold twice |
| 10+ | round(0.2n) | rest | all 5, evenly |

**`n >= 6` is about metric coverage, not validity.** Six is the smallest size
that appears in every fold; the co-presence guarantee holds identically at
`n = 3`. A group that is never validated still trains the model — it is simply
absent from the CV score, which is why metrics should be read by group size
rather than filtered by it.

"Singleton" means a group with one row *in the set being split*, so there are
two routes into it: a 1-row group in the whole table, and a 2-row group after
the holdout takes one. Such a row sits in the fit set of **all** k folds and the
validation set of **none** — normal k-fold inverts this. Validating it would
remove its only representative from that fold's fit set, which is the situation
the method exists to prevent.

## 4. Choosing a method

`random` is the default because only one of the three models can tell it from
`stratified_by_group`.

- **Only `BayesianConditionalModel` fits a per-group term** — a hierarchical
  random intercept (`models/bayesian_components.py:60-74`). `DirectCohortModel`
  and `IndependentTotalProbabilityModel` are structurally forbidden from seeing
  `neighborhood_id` as a feature (`modeling_config.py:830-840`). So co-presence
  is insurance for one model, not a requirement.
- **On the canonical population the guarantee is nearly vacuous.** Over 2000
  plain random 20% holdouts, 0.6% of draws leave a group with no training row,
  affecting 0.03 of 306 test rows. On the 60-neighborhood default config it is
  66% of draws and 1.56 of 50 rows — so prefer `stratified_by_group` there, and
  whenever the Bayesian model is the headline number.
- **Report `grouped` alongside, not instead.** The gap between `random` and
  `grouped` measures how much a model leans on group-specific signal. That
  matters because **a tree can identify a group implicitly**: the five
  neighborhood-level covariates are constant within a group and give 150 unique
  tuples for 150 neighborhoods — `(ses, median_age)` alone fingerprints each —
  so `DirectCohortModel` can reach a per-group intercept despite the ban. No
  structural check reveals this; only that gap does.
- **Under `grouped`, only a pooled or random effect is identified.** A fixed
  effect has no coefficient for an unseen level. The Bayesian model predicts
  such rows from the fitted population distribution and counts them in
  `_prediction_fallback_count` (`models/bayesian_conditional.py:388-395`).

**Do not filter small groups.** Dropping groups under 6 buildings costs 55% of
rows on the default config, biases the metric toward dense easy neighborhoods,
and removes exactly the small groups that partial pooling exists to handle.
Individual effects at `n = 1-2` are not a concern — pooling shrinks them to the
population mean. What *is* a concern is that `neighborhood_scale` is weakly
identified when most groups are tiny: a variance component needs replication
within groups. Canonical (min 4, median 10) is fine; the default config (10 of
60 at `n <= 2`) leans on its `HalfNormal(0.5)` prior.

## 5. Hazards

- **Draw the split once.** `train_test_split` re-draws on every call, so calling
  it again with a different `random_state` silently moves the test set. `cv` is
  the half meant to be re-derived freely. It requires an int `random_state`,
  so every `split()` call gives the same folds; tuning re-splits in every trial.
- **Give `cv` the training rows only** — `groups_train`, never `groups`. Hand it
  the whole table and the test set leaks into tuning.
- **Fold indices are positions inside `X_train`.** With a DataFrame the pandas
  index still identifies the original row; with a bare numpy array that link is
  gone.
- **Row order is an input to correctness, not just reproducibility.** Rows
  arrive sorted by neighborhood, so `KFold(shuffle=False)` would take contiguous
  blocks and `random` would behave as `grouped` — measured: 0-2 of ~13
  validation groups present in the fit set, against 31-36 of ~34 when shuffled.
  A split is a pure function of `(row order, random_state)`; sort by
  `building_id` at the call site if it must survive a reordering.
- **Cluster uncertainty by group under every method.** Rows in a group share its
  draw, so i.i.d. intervals are too narrow. `NeighborhoodClusterResampler`
  (`resampling.py:19-43`) already does this; effective *n* is nearer 150 than
  1529.

## 6. Modules and tests

| file | contents |
|---|---|
| `splitting/__init__.py` | public API: `Splitter`, `Method`, `StratifiedHoldout`, `StratifiedFolds` |
| `splitting/splitters.py` | `Splitter`, the `Method` literal, the type aliases, and the dispatch |
| `splitting/stratified.py` | `StratifiedHoldout`, `StratifiedFolds`, and the shared `_strata` helper |

Both custom classes implement `_iter_test_indices` and inherit `split` from
`BaseCrossValidator`, so the training half is always the complement — fit and
validation cannot overlap or miss a row, and a position returned by no fold is
fitted in every one. That is how singletons stay in training.

Tests: `tests/unit/test_splitters.py` (the methods and their guarantees) and
`tests/unit/test_splitting.py` (the two custom splitters).

## 7. What remains

1. **Migrate the consumers.** `experiment/partitions.py` shrinks rather than
   ports — its disjointness and exact-partition checks become structurally
   impossible once folds are index complements. Then `models/fold_scoring.py`
   and the two models' nested tuning loops.
2. **Report metrics by group size** (1 / 2-3 / 4-7 / 8+) instead of filtering,
   so the coverage gap in §3 is visible rather than silent.
3. **Surface `_prediction_fallback_count`** in the Bayesian model's evidence,
   turning the co-presence question into a number.
4. **Decide the write-once holdout guarantee's new home.** The
   exists-then-compare-else-write in `persist_split_manifest` is the one part of
   `data_splitting.py` worth keeping; it belongs wherever `artifacts/` is owned.
5. **Then delete** `data_splitting.py`, `OuterSplitConfig` and `FoldConfig`, and
   fold [Data and splitting](DATA_AND_SPLITTING.md) and
   [Cross-validation and selection](CROSS_VALIDATION_AND_SELECTION.md) into this
   document.

**Time-based splitting is deferred**, and not for want of design: there is no
time column anywhere. The simulator emits 16 columns, one row per building, a
single cross-section (`student_simulator/pipeline.py:15-32`), and its config
check is named `validate_cross_section_contracts`. That work starts at the
simulator. Three traps for whoever does it: `TimeSeriesSplit` splits on row
*position* rather than time value, so unbalanced periods land on both sides;
panel data breaks the row exchangeability every method here assumes; and time
*and* group together is an intersection of two constraints that one `groups`
channel cannot express.
