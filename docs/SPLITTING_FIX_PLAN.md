# fix(splitting): readability and a groups=None bug

## Context

Branch `fix/splitting` (off `feat/hyperparameter-tuning`; draft PR will target it). Probing the `splitting/` package found:

- **Silent bug:** `StratifiedHoldout.split(X, y, groups=None)` returns a full train set and an **empty test set with no error** — `np.asarray(None)` is a 0-d object array, `np.unique` yields `[None]`, that single "stratum" is skipped as a singleton, so nothing is held out. `StratifiedFolds` with `groups=None` raises, but with the misleading "0 rows have a stratum-mate" message.
- **API mismatch:** the evaluator ([evaluator.py:62-63](src/age_group_prediction/hyperparameter_tuning/evaluator.py#L62-L63)) documents `groups=None` for the `random` method, and `Splitter.cv("random")` wants no groups (KFold warns) — yet `Splitter.train_test_split` requires real groups even for `random` and dies with a raw `TypeError` in `take_rows(None, index)` otherwise. A `random` caller must fabricate groups just to drop them.
- **Readability drift:** [splitters.py:28](src/age_group_prediction/splitting/splitters.py#L28) `__all__` still re-exports `DesignMatrix`/`Groups`/`Target`, which moved to `utils.py` (commit bebd432; nothing imports them from here); a hard-to-read nested clamp in `StratifiedHoldout`; three stale statements in `docs/SPLITTING.md`.

**Contract chosen:** `groups=None` is legal exactly where the method ignores groups (`random` in `train_test_split`, returning `None` group pieces) and a clear `ValueError` everywhere else — sklearn's own wording, "The 'groups' parameter should not be None.", so all three methods fail uniformly (`grouped` already raises it via sklearn).

**Status (2026-09-27):** Task 1 committed; Task 2 implemented, awaiting review; resume at Task 3.

**Workflow:** per sub-task: implement → run that sub-task's gates → **STOP for your validation; you commit**. I never commit. Old `data_splitting.py` is out of scope.

---

## Task 1 — fix(splitting): reject groups=None in the stratified splitters

### 1.1 Failing tests first
[tests/unit/test_splitting.py](tests/unit/test_splitting.py), Failures section — two tests, `pytest.raises(ValueError, match="should not be None")`:
- `next(StratifiedHoldout(0.2).split(_X(), groups=None))` (today: silently empty test set)
- `next(StratifiedFolds(n_splits=3).split(_X(), groups=None))` (today: misleading message)

Run them, show they fail on current code for the documented reasons.

### 1.2 The fix in `_strata`
[stratified.py:19](src/age_group_prediction/splitting/stratified.py#L19) — one check fixes both classes, since both funnel through it:
- Signature: `_strata(groups: ArrayLike | None, rng: np.random.Generator)`.
- Raise `ValueError("The 'groups' parameter should not be None.")` when `groups is None`.
- Move `np.asarray(groups)` inside `_strata`; drop it at the two call sites (lines 66, 106).

### 1.3 Gates
`uv run ruff check` + `format` on the two changed files · `uv run mypy` · `uv run pytest tests/unit/test_splitting.py -W error`

**STOP — validate 1.1–1.3; you commit.**

---

## Task 2 — fix(splitting): let train_test_split take groups=None for random

### 2.1 Failing tests first
[tests/unit/test_splitters.py](tests/unit/test_splitters.py):
- `random` + `groups=None` → returns `groups_train is None` and `groups_test is None`; X/y pieces partition as usual (today: raw `TypeError`).
- Parametrized over `("stratified_by_group", "grouped")` + `groups=None` → `pytest.raises(ValueError, match="should not be None")`.
- Confirm existing tests that pass real groups for all three methods still pass untouched.

### 2.2 The fix in `train_test_split`
[splitters.py:62-97](src/age_group_prediction/splitting/splitters.py#L62-L97):
- Signature: `groups: Groups | None` (still required/positional — preserves the no-defaults guarantee); return type `tuple[DesignMatrix, DesignMatrix, Target, Target, Groups | None, Groups | None]`.
- `keys = None if self.method == "random" else groups` (line 77) stays as is.
- Guard the unpack: comprehension covers `X`/`y` only (four names), then `groups_train, groups_test = (None, None) if groups is None else (take_rows(groups, train_index), take_rows(groups, test_index))`.
- **No new None check for non-random methods**: `next(holdout.split(X, groups=None))` raises the Task-1 message from `StratifiedHoldout` and sklearn's identical message from `GroupShuffleSplit` — one source of truth.
- Docstring: one sentence — `groups` may be `None` only for `random`, which returns `None` for both group pieces; the other methods raise.

### 2.3 Gates
ruff check/format on the two changed files · `uv run mypy` · `uv run pytest tests/unit/test_splitters.py -W error`

**STOP — validate 2.1–2.3; you commit.**

---

## Task 3 — refactor(splitting): readability, no behavior change

### 3.1 Drop the stale re-exports
[splitters.py:28](src/age_group_prediction/splitting/splitters.py#L28): `__all__ = ["Method", "Splitter"]` (aliases live in `utils.py` since bebd432; verified nothing imports them from `splitters`).

### 3.2 Name the clamp
[stratified.py:59-64](src/age_group_prediction/splitting/stratified.py#L59-L64): extract into a module-level `_held_out_count(size: int, test_size: float) -> int` returning `max(1, min(round(size * test_size), size - 1))`, docstring "at least one row, never the whole stratum". The comprehension becomes `positions[: _held_out_count(positions.size, self.test_size)]`.

Deliberately left alone (surgical): the method dispatch in `train_test_split`/`cv` (plain if-chains with per-branch comments; `test_each_method_pairs_one_split_with_one_validator` monkeypatches the constructors and a table would break its premise); the two `np.concatenate(...) if ... else np.empty(...)` sites (class-specific comments); the `keys`/`label` names (commented; renaming churns the diff).

### 3.3 Gates
No new tests — the existing suites prove no behavior change. ruff · `uv run mypy` · `uv run pytest tests/unit/test_splitting.py tests/unit/test_splitters.py -W error`

**STOP — validate 3.1–3.3; you commit.**

---

## Task 4 — docs(splitting): correct SPLITTING.md

### 4.1 Fix the three stale statements
[docs/SPLITTING.md](docs/SPLITTING.md):
- Line 19: drop "(default)" from the `random` row — `Splitter` has no default method (line 53 of the doc itself says no parameter has a default).
- Line 24: replace "The other four are scikit-learn's own" with the named classes (`ShuffleSplit`, `KFold`, `GroupShuffleSplit`, `GroupKFold`).
- Line 150: remove "the type aliases" from the `splitters.py` row (they live in `utils.py`).

### 4.2 Document the new groups=None contract
§2: one sentence — `random` accepts `groups=None` and returns `None` group pieces; the group-aware methods raise `ValueError`.

### 4.3 Final gate on the branch
`uv run pytest -m "not slow"` (full suite, ~1002 tests) · `uv run ruff check` · `uv run mypy`.

**STOP — validate 4.1–4.3; you commit, then push and open the draft PR (base `feat/hyperparameter-tuning`) per your workflow.**

---

## Sequencing

Task 1 must precede Task 2 (Task 2's non-random error path relies on Task 1's raise). Tasks 3–4 follow in either order; docs last so they describe the finished contract.
