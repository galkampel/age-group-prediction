# feat(feature_engineering): center by a reference point, a public Log1pRatioScaler, explicit builder loops

## Context

Three small changes to `src/age_group_prediction/feature_engineering/`, requested 2026-09-27:

1. **A transformer that centers a column on a fixed reference point**, so the output measures deviation from it. The application is SES. The hypothesis is that both strong- and weak-SES populations have larger households than the middle, so the useful quantity is distance from the *true* SES mean, not from the fold mean. The docs should say what the SES reference point is.
2. **`_Log1pRatioScaler` becomes public** (`Log1pRatioScaler`).
3. **`FeatureTransformer._build_interactions` and `_build` use explicit `for` loops** instead of list comprehensions, for readability.

**Facts established while planning** (verified 2026-09-27; re-check before relying on them):
- `ses` is one continuous, neighborhood-level column drawn from `N(ses_mean=0.0, ses_sd=1.0)`, clipped to [-2.5, 2.5] (`configs/simulation.toml:14-17`, `student_simulator/neighborhood.py:78-93`). There are no strong/weak levels. "Strong" and "weak" are the two tails, which is why a deviation from the centre, squared, captures them both.
- **The SES reference point is 0.0**, the simulator's population mean. The sample mean is 0.07, and that is what `Center` (the fold mean) uses today (`docs/FEATURE_TRANSFORMATIONS.md` §3.2). The doc already argues for this: "If a substantive turning point is hypothesized, that value is a better $c$ than the mean."
- **In the simulator, SES does not affect household size.** `avg_household_size` depends on median age plus noise only (`neighborhood.py:112-133`); measured r(ses, household size) = −0.13. The U-shape the hypothesis describes is simulated in the **number of children**, through `ses_squared` (`simulation.toml:67` `ses_quadratic_coef = 0.05`; `outcomes.py:152-153`). The docs must state this, so the SES deviation feature is not presented as explaining household size in this data.
- The package imports nothing from the project (`transforms.py:23-24`), so the 0.0 is written at the call site, not read from the simulator config.

## Branch

- **Name:** `feat/feature-engineering-reference-point`
- **Base:** `feat/hyperparameter-tuning`, not `main`. `docs/FEATURE_TRANSFORMATIONS.md` differs by about 100 lines between the two, including §8, which this work edits. Basing on `main` would conflict once PR #5 merges. The code under `feature_engineering/` is identical on both.
- **Draft PR** into `feat/hyperparameter-tuning`, merged with a merge commit, the same as PR #7. If PR #5 is squash-merged into `main` first, rebase before retargeting: `git rebase --onto origin/main feat/hyperparameter-tuning feat/feature-engineering-reference-point`.

## Workflow (every task)

Plan the task, write failing tests first where behavior changes, then implement, run the gates, run a mutation check, run an independent review subagent, then **STOP**. The user reviews and commits. Never commit. Suggested commit messages carry no `Co-Authored-By` line. Each task is one commit.

Gates: `uv run ruff check <changed files>` + `uv run ruff format <changed files>` · `uv run mypy` (its `files` includes `feature_engineering`, with `disallow_untyped_defs`) · `uv run pytest tests/unit/test_feature_transforms.py tests/unit/test_feature_transformer.py tests/unit/test_transformer_spec.py -W error`.

## Task 0 — docs: commit this plan

This plan is saved as `docs/FEATURE_ENGINEERING_REFERENCE_POINT_PLAN.md`, written untracked on `feat/hyperparameter-tuning`. `git switch -c` carries an untracked file onto the new branch, so after creating the branch the user commits the plan as its first commit: `docs(feature_engineering): plan for the reference-point branch`.

## Task 0b — fix(feature_engineering): reject a log after centering or standardizing

Added 2026-09-28, after Task 1's baseline gate failed before any edit. Under `-W error`, `test_a_log_after_another_transform_is_caught_by_the_output_check` failed. It fitted `Center() → Log()`, and sklearn's `check_inverse` warned about the resulting `nan` before the finite-output check ran. The user ruled that a log must be rejected only where an earlier step makes negative values certain, and left to the data otherwise.

- **Rule:** each transform declares, as `ClassVar`s on `_TransformBase`, `negative_output` (`"always"` for `Center` and `Standardize`; `"as_input"` for `DomainScale`, `Log1p`; `"depends_on_data"` for `DomainMinMax`, `Log`, `RelativeSaturation`; `"never"` for `Quadratic`, `OneHot`) and `needs_nonnegative_input` (`Log`, `Log1p`, `RelativeSaturation`). Both are class-level, so dumped specs are unchanged.
- **Where:** `ColumnPlan._check_input_signs` (`transformer.py`) walks the chain and rejects a step needing non-negative input once negatives are certain. `"always"` makes them certain, `"as_input"` keeps that, `"depends_on_data"` and `"never"` clear it. Nothing else changes at runtime: a data-dependent bad value still fails at fit or transform, by the step's own input check (`RelativeSaturation`) or the finite-output check. The rule is on sign, so `Center → Log1p` is rejected even though `log1p` survives values in (-1, 0): a log of a mean-zero column is a mistake either way.
- **Tests:** rejection cases `center-log`, `standardize-log`, `standardize-log1p`, `center-relative_saturation`, plus `Center → DomainScale → Log`; accepted chains `DomainMinMax → Log`, `Log → Log`, `Center → DomainMinMax(-10, 10) → Log`, `Center → Quadratic → Log1p`, `Log → Center`; a bad value under `DomainMinMax → Log` fails with "non-finite"; and `_SIGN_BEHAVIOR` in `test_feature_transforms.py` pins every member's two facts, so a new transform must be classified.
- **Docs:** the `Log` docstring no longer claims a fit-time check that never existed, and `FEATURE_TRANSFORMATIONS.md` §8 lists the rule.

## Task 1 — refactor(feature_engineering): explicit loops in the builders (no behavior change)

In [transformer.py](src/age_group_prediction/feature_engineering/transformer.py):
- `_build` (lines 169-181): build the entry list with `for plan in self.plans: entries.append((plan.name, plan.build(), list(plan.columns)))`.
- `_build_interactions` (lines 183-198): start `entries` with the `("base", "passthrough", _all_columns)` entry, then `for interaction in self.interactions: entries.append(...)`.
- Keep both comment blocks. The one explaining why the base entry exists belongs above the base entry.
- Annotate the list explicitly, e.g. `entries: list[tuple[str, Any, Any]]`. Without the annotation, mypy infers the element type from the first entry: `(str, str, Callable)` in `_build_interactions`. Appending a transformer then fails type checking.

Tests: none new. The interaction and structure tests in `tests/unit/test_transformer_spec.py:298-443` and `test_feature_transformer.py` cover both builders.
Mutation check: drop the base entry, then any one interaction. Existing tests must fail each time.

## Task 2 — refactor(feature_engineering): make Log1pRatioScaler public

**Why it was private.** Nothing records a reason; the only commit is 9e84279. The design treats the pydantic specs as the public API and the sklearn objects as build details. It is the only hand-written estimator, which is the only reason it has a class name at all.

**Why make it public.** It is a real, reusable sklearn estimator with its own contract: it learns `mean_`, supports `inverse_transform`, and implements `get_feature_names_out`. Making it public lets it be imported for direct use and for `isinstance` checks. Nothing depends on it being private.

- [transforms.py](src/age_group_prediction/feature_engineering/transforms.py): rename `_Log1pRatioScaler` to `Log1pRatioScaler` in all three places, lines 178, 244 and 247.
- [feature_engineering/__init__.py](src/age_group_prediction/feature_engineering/__init__.py): import it and add it to `__all__`.
- [docs/MODULE_REFERENCE.md:72](docs/MODULE_REFERENCE.md#L72): add it to the `transforms.py` API list.
- Tests: add one test that `from age_group_prediction.feature_engineering import Log1pRatioScaler` works, and that `RelativeSaturation().build()` is an instance of it. The existing sklearn-contract block (`test_feature_transforms.py:242-306`) stays as is.
- Check: `git grep -n "_Log1pRatioScaler"` returns nothing.

## Task 3 — feat(feature_engineering): CenterByReferencePoint

In [transforms.py](src/age_group_prediction/feature_engineering/transforms.py), next to `DomainScale` and `DomainMinMax` (the "fixed, from domain knowledge" family):

```python
class CenterByReferencePoint(_TransformBase):
    """Subtract a fixed reference point chosen from domain knowledge.

    Unlike ``Center``, it learns nothing from the data, so the output means
    "deviation from the reference" in every fold.
    """

    kind: Literal["center_by_reference_point"] = "center_by_reference_point"
    reference_point: float

    def build(self) -> TransformerMixin:
        reference_point = self.reference_point
        return FunctionTransformer(
            lambda x: x - reference_point,
            inverse_func=lambda x: x + reference_point,
            validate=False,
            feature_names_out="one-to-one",
        )
```

- `reference_point` has no default. Like every parameter in this package, it is a decision the call site states.
- `feature_names_out="one-to-one"` keeps the column name, as `Center` does.
- Add it to the `Transform` discriminated union (`transforms.py:230-241`) and to the package `__all__`.
- Update the `Quadratic` docstring ("Meant to follow `Center` or `Standardize`") to also name `CenterByReferencePoint`.
- Declare `negative_output = "depends_on_data"` on it (Task 0b): a column can lie entirely above the reference. Add `"center_by_reference_point": ("depends_on_data", False)` to `_SIGN_BEHAVIOR` in `test_feature_transforms.py`.

**Tests first** (hand-written expected values, following the existing style):
- In `test_feature_transforms.py`:
  - `[1.0, 2.0, 3.0]` with `reference_point=2.0` gives `[-1.0, 0.0, 1.0]`, and `inverse_transform` round-trips.
  - It learns nothing: fit on one frame, transform another, and the output is `x - reference_point` regardless of what it was fitted on.
  - Omitting `reference_point`, or a typo such as `reference=0.0`, raises a pydantic `ValidationError`.
  - The column name is unchanged (`get_feature_names_out`).
- Add it to `_EVERY_MEMBER` (`test_feature_transforms.py:32-42`, the serialization round-trip) and to `_ONE_OF_EACH_TRANSFORM` (`test_transformer_spec.py:39-49`).
- The application, in `test_feature_transformer.py`: `ColumnPlan(name="ses", columns=("ses",), transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic()))` on a hand-chosen frame emits `[ses, ses²]` measured from 0, not from the frame's mean.

Mutation check: change `x - reference_point` to `x + reference_point`, and replace `reference_point` with the fold mean. Tests must fail each time.

## Task 4 — docs(feature_engineering): the SES reference point

In [docs/FEATURE_TRANSFORMATIONS.md](docs/FEATURE_TRANSFORMATIONS.md), re-read in full first:
- §3.2 (SES candidates) and §4.2 (`ses` for the GLMs): state that **the SES reference point is 0.0**, the simulated population mean. Contrast it with the fold mean of 0.07 that `Center` uses. Explain why a fixed reference is better here: the same meaning in every fold, and it is the point the hypothesis names. State the hypothesis, that strong- and weak-SES populations have larger households, and say plainly that the simulator puts this U-shape in the number of children, not in `avg_household_size`.
- §8 (building each model's transformer): add `CenterByReferencePoint` to the import block, and switch the quadratic SES declaration(s) from `Center()` to `CenterByReferencePoint(reference_point=0.0)`, if §3.2's reasoning then recommends it for that model.
- §7 summary table: update the `ses` row.
- §1 "Prefer fixed anchors over learned scales": list the new transform alongside `DomainScale` and `DomainMinMax`.
- Wherever the doc names `RelativeSaturation`'s estimator, use `Log1pRatioScaler`.

In [docs/MODULE_REFERENCE.md:72](docs/MODULE_REFERENCE.md#L72), add `CenterByReferencePoint` to the API list.

## Final gate (after Task 4)

- `uv run pytest -m "not slow"` (1007 passed on the base as of 2026-09-27, plus this branch's new tests).
- ruff on the changed files · `uv run mypy`.
- `git grep -n "_Log1pRatioScaler"` returns nothing.
- Independent review of the docs against the code.
- At the stop: the commit command, the draft PR description, and the push command.

## Sequencing

Task 0b went first: it unblocked the `-W error` gate. Task 1 comes next: it is the smallest and touches only `transformer.py`. Tasks 2 and 3 both edit `transforms.py` and `__init__.py`, so do them one at a time. Docs go last, so they describe the finished API.
