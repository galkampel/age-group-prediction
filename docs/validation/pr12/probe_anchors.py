"""PR #12 validation V4: every markdown link (#slug and file.md#slug) in the edited docs resolves, GitHub slug rules."""

import re
from pathlib import Path

FILES = [
    "docs/DIRECT_COHORT_MODEL.md",
    "docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md",
    "docs/FEATURE_TRANSFORMATIONS.md",
    "docs/MODULE_REFERENCE.md",
    "docs/HYPERPARAMETER_TUNING_PLAN.md",
    "docs/DIRECT_COHORT_GENERALIZATION_PLAN.md",
    "docs/DIRECT_COHORT_GENERALIZATION_HANDOFF.md",
    "docs/DIRECT_COHORT_GENERALIZATION_VALIDATION.md",
    "docs/MULTI_COHORT_MODELS_PLAN.md",
    "docs/MODEL_REIMPLEMENTATION_PLAN.md",
    "docs/README.md",
]


def slug(heading: str) -> str:
    h = re.sub(r"`", "", heading.strip())
    h = h.lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def slugs(path: Path) -> set[str]:
    out: dict[str, int] = {}
    fence = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fence = not fence
        if fence or not re.match(r"#{1,6} ", line):
            continue
        s = slug(re.sub(r"^#+ ", "", line))
        n = out.get(s, 0)
        out[s] = n + 1
        if n:
            out[f"{s}-{n}"] = 1
    return set(out)


bad = 0
total = 0
for f in FILES:
    path = Path(f)
    text = path.read_text()
    for m in re.finditer(r"\]\(([^)\s]+?\.md)?#([^)\s]+)\)", text):
        total += 1
        target = (path.parent / m.group(1)) if m.group(1) else path
        line = text[: m.start()].count("\n") + 1
        if not target.exists():
            print(f"MISSING FILE {f}:{line} -> {m.group(0)}")
            bad += 1
            continue
        if m.group(2) not in slugs(target):
            print(f"BAD ANCHOR {f}:{line} -> {m.group(0)}")
            bad += 1
    for m in re.finditer(r"\]\(([^)\s#]+?\.md)\)", text):
        total += 1
        if not (path.parent / m.group(1)).exists():
            print(
                f"MISSING FILE {f}:{text[: m.start()].count(chr(10)) + 1} -> {m.group(0)}"
            )
            bad += 1
print(f"{total} links checked, {bad} broken")
