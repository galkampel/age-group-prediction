# Hyperparameter tuning review and continuation

## Scope

Reviewed `1f4ed67..80fd068` on `feat/hyperparameter-tuning`, using Optuna
4.9.0. The package tests passed: `141 passed in 4.98s`, with
`uv run pytest tests/unit/test_hyperparameter_tuning_*.py -q -W error`.
Repository source/tests were not edited; mutations used a scratch copy.

This is review input for A4b, not an approved design. The source of truth is
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md). D12 remains always
minimize, with evaluator negation only for greater-is-better metrics. The
three highest-value items are A1, C1 and D1. No wrong winner or sign was
reproduced in the unmodified implementation.

## Findings

Reproduce every item against current code before accepting or rejecting it.
Cost figures are existing footprints, not net deletions; groups overlap.
Tests for retained guarantees must be kept or adapted, not blindly removed.

| ID | Finding and recorded evidence | Qualification or choice |
|---|---|---|
| A1, bug | Evaluator `Field(min_length=1)` rejects the empty search space required by D4's one-trial NB2 run. Output: `Value should have at least 1 item after validation, not 0`. Removing only that constraint in scratch produced `params = {}; trials = 1; folds = 2; state = COMPLETE`. | Verify with the actual planned NB2 model before implementing the fix. |
| A2, bug | Mutable categorical values are shared despite copying params dictionaries. Appending through best params, record params or `to_dict()['best_params']` changed the retained study to `{'x': [1, 2, 99]}`. | Decide with C3 whether to narrow choices or support nested values with a copying contract. |
| B1, should-fix | A five-fold fixed-predictor experiment gave actual MAE SE `0.06076`, RMS reported SE `0.09033`, ratio `1.487`. | Distinguish the split-size identity from exact sampling uncertainty. This does not justify changing the implemented correction formula. |
| B2, simplification | D7 explains why not to wrap Optuna objects, but not which planned client needs configurable sampler/pruner fields. | Compare their use separately from keeping seeded TPE and no-pruning defaults. Overlaps C1. |
| C1, simplification | Remove unused timeout/jobs/initial-params/custom sampler-pruner options, associated guards/callbacks and reproducibility flag: 42 attributable source lines; 13 test functions, 16 cases, 102 test lines. | Broader than S1's estimated 27 lines/~8 tests. E12 excludes jobs/timeouts and the notebook uses no pruning. Preserve COMPLETE-only selection and records for objectives still supported. |
| C2, simplification | Merge score/value pairs and remove per-record metric name: about nine declaration/plumbing lines; four tests need adaptation. | Compare with S2/S3. E9 uses best value and SE. Retain evaluator negation and any supported natural-sign reporting; `_sign` is not automatically redundant. |
| C3, simplification | Restrict categorical choices to planned scalars; remove the object-warning exemption: six suppression lines; three cases/20 test lines. | D4's calibration choices are scalar. Otherwise retain object support and resolve A2/E2 explicitly. |
| C4, simplification | Remove or relocate `to_dict`/`_record_dict`: 19 source lines; two tests/nine lines. | D4 specifies a logging helper and artifacts, without an explicit call to this method. State-name and tuple conversions remain useful; relocation must not merely add duplication. |
| C5, simplification | Optional aggregation policies/custom-subclass support: ABC 10 lines, Mean seven, UpperBound 13; shared policy cases affected. | No PR D run selects alternatives, but E9 mentions UpperBound. Confirm planned use; the existing preference to keep Mean is not an approved reversal. |
| C6, simplification | `_check_folds`: ten lines; six cases/eight test lines. NumPy already raises for empty/unequal weighted inputs. | Mean ignores sizes, so do not remove its different length check if Mean stays. Lost guarantee may be only custom messages for weighted methods. |
| C7, simplification | Merge `_weighted_mean`, `_create_optuna_study`, `_check_exposure`: function footprints three/12/eight lines. | These are not net savings after inlining. Preserve checks/defaults and readability. No notebook calls these private helpers. |
| C8, simplification | Reconsider splitter-only tests (four cases/eight lines plus ten-line helper), smoke/export tests (two cases/39 lines), setter-error delegation (one case/three lines). | Prove retained coverage catches the package mistake before deleting or moving tests. Line counts alone are not evidence of redundancy. |
| D1, should-fix | Scratch mutation `TPESampler(seed=self.seed, multivariate=True)` to `TPESampler(seed=0, multivariate=True)` still gave `141 passed`. | Same-seed repetition does not verify supplied-seed wiring. Strengthen an existing test. |
| D2, should-fix | Scratch mutation `best_score=best.values[0]` to `best_score=complete[0].values[0]` still gave `141 passed`. | Remove the field with S2 or test its consistency with a non-first winner. |
| E1, should-fix | A simpler model's mean `1.03` is within the best model's one-SE threshold `1.05`, but UpperBound scores `1.23` versus `1.05` and selects the complex model. | Describe an uncertainty penalty, not the classical one-SE simplicity rule. No ranking change is approved. |
| E2, should-fix | Tuple choices gave JSON round-trip equality `False`; object choices raised `TypeError: Object of type object is not JSON serializable`. | Qualify JSON safety or narrow choices. State/fold conversions do not recursively normalize params. |

Additional test qualification: a naive `s / sqrt(K)` mutation left the
record-SE and splitter-ratio selection at `5 passed, 21 deselected`, but the
independent hand-value test failed: `0.6009252125773314 != 0.9501461875826149`.
The package does have a formula regression guard. Advancing a supplied seeded
sampler is not itself a bug: fresh samplers are promised only for the default.
The original NaN-category probe did not establish a reported bug.

## Reproduction

The original `/tmp/tuning-review-80fd068/` scripts are optional and may be
cleared. These read-only snippets run using the repository `.venv` with
`PYTHONPATH=src`. Their outputs describe `80fd068`, not permanent assertions
about later code.

```python
import json
import warnings
import optuna
from sklearn.linear_model import PoissonRegressor
from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator, CategoricalParameter, HyperparameterStudy,
)
from age_group_prediction.modeling import CountModel
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter

optuna.logging.set_verbosity(optuna.logging.ERROR)
try:
    CVHyperparameterEvaluator(
        CountModel(estimator=PoissonRegressor()), [],
        cv=Splitter("random").cv(n_splits=2, random_state=0),
        metric=POISSON_DEVIANCE,
    )
except ValueError as error:
    print("EMPTY:", error.errors()[0]["msg"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    for choices in (([1, 2],), ((1, 2),), (object(),)):
        parameter = CategoricalParameter("choice", choices)
        study = HyperparameterStudy(seed=0, n_trials=1)
        result = study.optimize(lambda trial: (parameter.suggest(trial), 1.0)[1])
        if isinstance(choices[0], list):
            result.best_params["choice"].append(99)
            print("NESTED:", study.optuna_study_.trials[0].params)
        else:
            payload = result.to_dict()
            try:
                print("ROUNDTRIP:", json.loads(json.dumps(payload)) == payload)
            except TypeError as error:
                print("JSON:", str(error))
```

Output: `EMPTY: Value should have at least 1 item after validation, not 0`;
`NESTED: {'choice': [1, 2, 99]}`; `ROUNDTRIP: False`;
`JSON: Object of type object is not JSON serializable`.
Repeat the list case independently through record params and serialized
best params to reproduce the other A2 paths.

```python
import numpy as np
from age_group_prediction.hyperparameter_tuning import UpperBound, corrected_std_error

rng = np.random.default_rng(7)
losses = np.abs(rng.normal(size=(20000, 5, 20)))
scores = losses.mean(axis=2)
actual = scores.mean(axis=1).std(ddof=1)
reported = np.array([corrected_std_error(fold) for fold in scores])
print("SE:", round(actual, 5), round(np.sqrt(np.mean(reported ** 2)), 5))
simple = [1.03 - 0.2 / np.sqrt(3), 1.03 + 0.2 / np.sqrt(3)]
complex_scores = [1.0 - 0.05 / np.sqrt(3), 1.0 + 0.05 / np.sqrt(3)]
print("BOUND:", round(UpperBound().aggregate(simple, [1, 1]), 2),
      round(UpperBound().aggregate(complex_scores, [1, 1]), 2))
```

Output: `SE: 0.06076 0.09033`; `BOUND: 1.23 1.05`. These MAE vectors
represent a fixed predictor on 100 independent validation rows, covered once.

For D1/D2, create a fresh scratch copy of `src`, `tests`, `configs` and
`pyproject.toml` from `git archive 80fd068`. Use the original `.venv` interpreter,
verify the imported module's path is under scratch, apply one mutation from
the table, and run `pytest tests/unit/test_hyperparameter_tuning_*.py -q -W error`
with `PYTHONPATH=src` from that copy. Restore scratch files by copy and `cmp`
between mutations. Repeat on current-code copies for current verdicts; never
mutate repository source during the review. Recalculate net simplification
costs from current definitions and tests rather than trusting old footprints.

## Continuation prompt

```text
Read docs/EXPERIMENT_NOTEBOOK_PLAN.md: sections 1-4 and 11 fully, and in
section 6 the status of A0-A4, A4b and A5. It is the source of truth for my
workflow (section 3) and settled E1-E12; do not re-ask those decisions.
Then read docs/HYPERPARAMETER_TUNING_PLAN.md sections 2, 4, 6 and 7,
docs/HYPERPARAMETER_TUNING_REVIEW.md fully, and the package
src/age_group_prediction/hyperparameter_tuning/ with its tests
tests/unit/test_hyperparameter_tuning_*.py. D12 is always minimize.

Expected branch: feat/hyperparameter-tuning, draft PR #5. A0-A4 were
committed through 80fd068. Verify actual HEAD and worktree before assuming
that A4b is committed or the tree is clean. Preserve user changes.
I commit and push myself; never commit or add Co-Authored-By lines.

The review document replaces the pasted report. Observations and cost
estimates are input, not accepted changes. Temporary scripts are optional.

Start A4b in plan mode, with no repository source/test edits:
- Reproduce every finding A1-E2 against current code using read-only probes
  or isolated scratch mutations. Verify imports, restore scratch files by
  copy and cmp, show commands/output, and give accept/reject verdicts with
  qualifications. Identify already-fixed findings separately.
- Check empty-space evaluation against D4's actual one-trial NB2 client.
  Couple object-choice aliasing and JSON findings with the choice of whether
  to retain arbitrary objects. Preserve D12 and any retained natural-sign
  reporting. Do not change the SE formula merely to clarify its description.
- Compare C1-C8 with A4b's S1-S3. Separate narrow S1 from sampler/pruner
  removal. E9 mentions UpperBound; establish planned use before removing
  policies or serialization. Footprints overlap and are not net savings;
  preserve tests for retained guarantees.
- Strengthen supplied-seed wiring coverage. For best_score, either remove
  the field with S2 or verify a non-first winner's score. Reuse existing tests.
- Rank recommendations: what goes, net lines/tests affected, guarantee lost,
  and planned-client reference. Explain verdicts differing from the report.
  Ask me to choose, then STOP.

Only after my approval, implement the chosen changes with section 3's full
routine. Update the tuning plan's decisions, design and test table to match;
document the empty-space, copying/JSON and statistical contracts that remain.
End with a file-by-file summary, checks and suggested commit commands.
Do not start A5 or freeze its user guide around an unsettled API.

If a material plan/code assumption fails, stop and show the difference and
evidence before changing scope. A recorded finding or historical test count
is not an invariant of the current checkout.
```