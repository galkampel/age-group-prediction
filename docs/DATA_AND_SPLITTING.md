# Data And Splitting

This document covers everything between the simulator's building table and the
first model fit: the validated modeling table, the known-neighborhood outer
split that creates the lockbox, the persisted split manifest and its replay,
the training-only validation folds, and the partition checks the experiment
runner performs before any candidate is fitted. Everything here is derived
from `src/age_group_prediction/dataset_builder.py`, `data_splitting.py`,
`hashing.py`, `ModelingSchema`/`OuterSplitConfig`/`FoldConfig`/`RandomnessConfig`
in `modeling_config.py`, `experiment_config.py`, `experiment/partitions.py`, and
the manifest helpers in `experiment/evidence.py`.

## 1. Deployment Claim

Every split in the project evaluates one claim: **a new building in a known
neighborhood**. It is the default value of `SplitManifest.deployment_claim`
and is copied into cross-validation evidence. Two consequences follow:

- a holdout or validation building always has at least one building from its
  own neighborhood in the rows used for fitting;
- a neighborhood with a single building cannot be evaluated under this claim,
  so it stays wholly in training.

An "unseen neighborhood" claim would need a different split strategy and a
new `strategy_version`; none is implemented.

## 2. The Modeling Table

`build_modeling_table(source_df, *, schema=DEFAULT_MODELING_SCHEMA)` returns a
new DataFrame. In order it:

1. validates the schema definition itself (`schema.validate_definition()`);
2. refuses a source missing any ID, raw numeric, categorical, room-count, or
   target column (`ValueError("Missing source columns: [...]")`);
3. copies the base columns;
4. derives each room share as `source[room] / source["n_apartments"]`;
5. reorders to `schema.table_columns` and calls `schema.validate_table`.

Raw room counts are not kept in the output.

| Role | Columns (canonical order) |
|---|---|
| IDs | `building_id`, `neighborhood_id` |
| Raw numeric features | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, `n_apartments` |
| Derived room shares | `3_rooms_share`, `4_rooms_share`, `5_rooms_share` (`6_rooms` is the reference, not emitted) |
| Categorical feature | `school_status` (`none`, `existing`, `planned`; reference `none`) |
| Cohort targets | `n_kindergarten`, `n_elementary`, `n_highschool` |
| Total target | `n_children_total` |

`n_apartments` is the exposure column. The schema version is `"1"`.

### Validation

`ModelingSchema.validate_definition` raises `ValueError` when ID columns
coincide, the exposure is not a raw numeric feature, room columns repeat or
the reference is not a room column, the number of derived shares is not one
fewer than the number of room columns (names are not compared), categorical or table columns repeat, or a feature is
forbidden. The forbidden feature columns are the simulator's latent effects
`u_b`, `w_j`, and `v_b`; IDs and targets are also refused as features. The
default schema is validated at import time.

`ModelingSchema.validate_table` first re-validates the definition, then raises
`ValueError` for:

| Check | Message (abridged) |
|---|---|
| Every table column present | `Missing modeling columns` |
| `building_id` unique | `building_id must be unique` |
| Exposure strictly positive | `n_apartments must be positive` |
| Numeric features and targets finite | `Modeling numeric columns must be finite` |
| Targets nonnegative integers (integral floats pass) | `Modeling targets must be nonnegative integers` |
| Categorical values present and known | `... must not be missing`, `Unknown ... values` |
| Cohorts sum exactly to the total | `n_children_total must equal the sum of cohort targets` |

## 3. Outer Split Algorithm

```text
split_known_neighborhood_buildings(
    modeling_table, *, config=DEFAULT_OUTER_SPLIT_CONFIG, rng=None,
) -> DataSplit(train_df, test_df, manifest)
```

The split checks only that the two ID columns exist and that building IDs are
unique (`Missing split columns`, `building_id must be unique before
splitting`). It does **not** call `validate_table`; build the table with
`build_modeling_table` first.

For each neighborhood $j$ with $n_j$ buildings, processed in sorted
neighborhood order with building IDs sorted:

- if $n_j = 1$, the building stays in training and **no random draw is made**;
- otherwise the holdout count is

$$
h_j = \max\bigl(1,\ \min(\operatorname{round}(n_j f),\ n_j - 1)\bigr),
$$

  where $f$ is `test_fraction`, and the holdout is the first $h_j$ IDs of
  `rng.permutation(sorted_ids)`.

`round` is Python's built-in, which rounds halves to even: $n_j = 5$ with
$f = 0.5$ holds out 2, and $n_j = 10$ with $f = 0.25$ also holds out 2.
Sorting before permuting makes the split independent of input row order.

**RNG.** `rng=None` uses `np.random.default_rng(DEFAULT_SEED)` with
`DEFAULT_SEED = 42`. Only the source is recorded, never a seed value:
`seed_source` is `"project_default"` when `rng` is `None` and
`"caller_generator"` otherwise.

**Requested vs realized fraction.** `requested_holdout_fraction` is
`test_fraction`; `realized_holdout_fraction` is
$|\text{holdout}| / n_{\text{rows}}$. The clamp pushes the realized fraction up
in small neighborhoods (every non-singleton contributes at least one), and
singletons pull it down.

## 4. `SplitManifest`

An immutable dataclass; `to_dict()` is `dataclasses.asdict`.

| Field | Type | Content |
|---|---|---|
| `strategy` | `str` | `config.strategy_version` (`known-neighborhood-v1`) |
| `requested_holdout_fraction` | `float` | `test_fraction` |
| `realized_holdout_fraction` | `float` | holdout rows / all rows |
| `training_building_ids` | `tuple` | sorted training IDs |
| `holdout_building_ids` | `tuple` | sorted holdout IDs |
| `source_table_hash` | `str` | `table_hash` of the **full** modeling table |
| `column_schema_hash` | `str` | `column_schema_hash` of the full table |
| `n_rows` | `int` | rows in the full table |
| `n_neighborhoods` | `int` | distinct neighborhoods |
| `seed_source` | `SeedSource` | `project_default` or `caller_generator` (other values refused) |
| `deployment_claim` | `str` | default `new building in a known neighborhood` |

`SplitManifest.from_dict(payload)` refuses unknown or missing keys
(`Split manifest payload keys do not match the manifest schema`), except that
`deployment_claim` may be absent and then takes its default.

**Hashes** (`hashing.py`):

- `table_hash(df, *, id_column)` sorts by `id_column`, hashes each row with
  `pandas.util.hash_pandas_object(index=False)`, and returns SHA-256 of the
  row hashes. Row order does not matter; column order and dtypes do. It does
  not check ID uniqueness.
- `column_schema_hash(df)` is SHA-256 of the JSON list of
  `(column name, dtype string)` pairs in column order.

**Fingerprint.** The experiment layer identifies a manifest by
`_manifest_fingerprint`: SHA-256 of `json.dumps(manifest.to_dict(),
sort_keys=True, separators=(",", ":"))`. Freezes, provenance, and the final
evaluation compare this value. `_split_summary` records every manifest field
except the ID lists, which it replaces with `training_building_count` and
`holdout_building_count`, so logged provenance contains no building IDs.

## 5. Persistence And Replay

```text
persist_split_manifest(path, manifest) -> SplitManifest
load_split_manifest(path) -> SplitManifest
replay_split_manifest(modeling_table, manifest, *, config=DEFAULT_OUTER_SPLIT_CONFIG) -> DataSplit
```

**Write-once persistence.** If `path` exists, `persist_split_manifest` loads
it and compares the parsed manifests field by field. An unequal manifest
raises `ValueError("A different split manifest is already persisted at ...;
refusing to overwrite the lockbox partition")`; an equal one returns the
**existing** manifest without writing. Otherwise it creates parent
directories and writes `json.dumps(..., indent=2, sort_keys=True)`. The
exists-then-write sequence is not atomic, so use a single writer.

**Loading.** `load_split_manifest` is `json.loads` plus `from_dict`; it raises
`FileNotFoundError`, `json.JSONDecodeError`, or the `from_dict` `ValueError`.

**Replay** rebuilds the exact split from the table. It raises `ValueError`, in
this order, when:

1. the ID columns are missing or building IDs are duplicated;
2. `table_hash` differs (`Modeling table does not match the split manifest`);
3. `column_schema_hash` differs;
4. `manifest.strategy != config.strategy_version`;
5. `n_rows` or `n_neighborhoods` differ;
6. training and holdout IDs overlap or do not together equal the table's IDs;
7. the recomputed realized fraction is not `np.isclose` to the manifest's
   (this catches swapped partitions).

Replay does **not** check `seed_source`, `requested_holdout_fraction`,
`deployment_claim`, or whether each neighborhood's holdout count follows the
formula in Section 3.

## 6. Training-Only Validation Folds

```text
make_validation_folds(train_df, *, config=DEFAULT_FOLD_CONFIG, rng=None) -> FoldPlan
FoldPlan(folds: tuple[ValidationFold, ...], unvalidated_building_ids: tuple)
ValidationFold(fold_index: int, fit_df: DataFrame, validation_df: DataFrame)
```

**Rotation.** With $k$ = `n_folds`, neighborhoods are visited in sorted order
and singletons are skipped. For a neighborhood's permuted sorted IDs,
building at position $i$ gets fold

$$
\text{fold} = (\text{cursor} + i) \bmod k,
\qquad
\text{cursor} \leftarrow (\text{cursor} + n_j) \bmod k .
$$

The cursor carries over between neighborhoods, so assignments run through
$0,\dots,k-1$ cyclically across the whole partition. Consequences:

- every non-singleton training building is validated **exactly once**;
- fold sizes differ by at most one building;
- a neighborhood's buildings occupy consecutive folds, so each validation
  building keeps at least one neighbor in that fold's `fit_df`.

There is no validation fraction; the share is about $1/k$.

`unvalidated_building_ids` lists, sorted, the singleton buildings that are
always in `fit_df` and never validated. A fold with no validation rows raises
`ValueError("Fold {i} has no validation rows; reduce n_folds ({k}) or supply
more buildings per neighborhood")`. This happens only when there are fewer
non-singleton training buildings than folds.

`make_validation_folds` uses the same `rng=None` fallback as the outer split
but records no seed source; fold identity is recorded instead by fingerprint
(Section 7). The same function builds the models' internal tuning folds.

## 7. Partition Checks Before Modeling

`validate_experiment_partitions(outer_train_df, *, split_manifest,
validation_folds, schema=DEFAULT_MODELING_SCHEMA)` is the first step of
`run_cross_model_validation`. It raises `ValueError` when:

1. `schema.validate_table(outer_train_df)` fails;
2. no validation fold is supplied;
3. training IDs differ from `split_manifest.training_building_ids`;
4. the training frame contains holdout IDs;
5. fold indices repeat;
6. a fold's fit and validation IDs overlap;
7. a fold's fit and validation IDs do not together equal the training IDs;
8. a fold contains holdout IDs;
9. a fold frame has duplicate IDs, different columns, or rows not equal to the
   canonical training rows (`pandas.testing.assert_frame_equal`).

It returns one `FoldIdentity(fold_index, fit_building_ids,
validation_building_ids, fingerprint)` per fold, sorted by index. The
fingerprint is SHA-256 of `repr((fold_index, sorted fit IDs, sorted validation
IDs))`, where IDs are sorted by `(type name, repr)`, so integer IDs sort as
text. These identities are stored in the selection freeze.

## 8. The Canonical Lockbox

The canonical manifest is `artifacts/lockbox/split_manifest.json`. The path is
not a package constant; it is built only in `notebooks/02_model_fitting.py`,
which loads the manifest if it exists, otherwise splits with
`np.random.default_rng(randomness.default_seed)` and persists it, and in both
cases replays it to obtain the training partition. `artifacts/lockbox/` is
git-ignored.

Rules:

- **Never regenerate or overwrite it.** Results, freezes, and the final
  evaluation are bound to its fingerprint.
- **Never replay it outside the guarded workflow.** Replay materializes the
  holdout rows.
- The one-time rule is enforced by the write-once persistence above, by
  `SelectionFreeze.with_test_metrics` refusing a second test result, and by
  the tracking guards in [FINAL_EVALUATION.md](FINAL_EVALUATION.md).

Tests never touch it: the notebook characterization test runs in a copied
project directory.

## 9. Configuration

`load_experiment_config` requires every top-level section and refuses unknown
sections. Within these three sections, a missing key falls back to the
dataclass default and an unknown key raises `ValueError`.

| TOML section / field | Default | Effect |
|---|---|---|
| `[outer_split] test_fraction` | 0.2 | $f$ in Section 3; must satisfy $0 < f < 1$ |
| `building_id_column`, `neighborhood_id_column` | `building_id`, `neighborhood_id` | Split, hash, and replay keys; must differ |
| `strategy_version` | `known-neighborhood-v1` | Stored as `manifest.strategy`; checked on replay |
| `[folds] n_folds` | 5 | $k$ in Section 6; must be at least 2. Also Model B's internal `tuning_folds` |
| `building_id_column`, `neighborhood_id_column` | `building_id`, `neighborhood_id` | Fold keys; must differ |
| `[randomness] default_seed` | 42 | Master seed for cross-validation; the notebook also seeds the split and fold generators with it; must be nonnegative |

## 10. Pitfalls

- **Do not hand-assemble the table.** The split does not validate targets or
  features; `build_modeling_table`, the runner's partition checks, the final
  refit, and the lockbox evaluation do.
- **Dtype and column-order changes break replay.** Casting an integer column
  to float changes `table_hash` even when values are equal.
- **Expect the realized fraction to differ from 0.2.** Half-to-even rounding,
  the clamp, and singletons all move it.
- **A matching seed is not a matching split.** Only the seed source is
  recorded; replay verifies content, not randomness.
- **Singletons are never validated.** Check `unvalidated_building_ids` when
  interpreting fold coverage.
- **Stale comments.** Comments in `experiment/partitions.py` and
  `experiment/aggregation.py` describe folds as "repeated overlapping splits";
  the rotation above validates each building exactly once.

## 11. Tests And Related Documents

- Tests: `tests/unit/test_modeling_data.py` (table, split, manifest,
  persistence, replay, folds), `tests/unit/test_experiment.py` (partition
  checks and fold coverage), `tests/unit/test_modeling_config.py` (config
  loading), `tests/unit/test_final_evaluation.py` (manifest fingerprint and
  replay during final evaluation), and
  `tests/characterization/test_model_fitting_notebook.py` (notebook in a
  copied project).
- [MODELING_GUIDE.md](MODELING_GUIDE.md) Sections 2–4,
  [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md),
  [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md),
  [FINAL_EVALUATION.md](FINAL_EVALUATION.md).
