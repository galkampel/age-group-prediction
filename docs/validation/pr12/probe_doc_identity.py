"""PR #12 validation V2: DIRECT_COHORT_MODEL.md §0.1 contains plan §2b word for word, from "**Setup.**" to the end of (E)."""

import difflib
from pathlib import Path


def section(path: str) -> str:
    text = Path(path).read_text()
    start = text.index("**Setup.**")
    end_marker = "(B) for every estimator instead."
    end = text.index(end_marker, start) + len(end_marker)
    return text[start:end]


plan = section("docs/DIRECT_COHORT_GENERALIZATION_PLAN.md")
doc = section("docs/DIRECT_COHORT_MODEL.md")
if plan == doc:
    print(f"IDENTICAL: {len(plan)} characters, {plan.count(chr(10)) + 1} lines")
else:
    diff = list(
        difflib.unified_diff(
            plan.splitlines(),
            doc.splitlines(),
            "plan §2b",
            "doc §0.1",
            lineterm="",
            n=0,
        )
    )
    print(f"DIFFER: {len(diff)} diff lines")
    print("\n".join(diff[:60]))
