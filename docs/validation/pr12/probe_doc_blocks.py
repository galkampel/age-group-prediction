"""PR #12 validation V4: run every python block of FEATURE_TRANSFORMATIONS.md §8.0–§8.3,
DIRECT_COHORT_MODEL.md §0 and INDEPENDENT_TOTAL_PROBABILITY_MODEL.md §0, in order, in one
namespace, under warnings-as-errors."""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from age_group_prediction.splitting import Splitter
from student_simulator.config import load_simulation_config
from student_simulator.pipeline import StudentPopulationSimulator

warnings.simplefilter("error")


def blocks(path: str, start: str, stop: str) -> list[tuple[int, str]]:
    text = Path(path).read_text()
    lo = text.index(start)
    hi = text.index(stop, lo)
    section = text[lo:hi]
    offset = text[:lo].count("\n")
    out = []
    for m in re.finditer(r"```python\n(.*?)```", section, re.DOTALL):
        line = offset + section[: m.start()].count("\n") + 1
        out.append((line, m.group(1)))
    return out


SECTIONS = [
    (
        "docs/FEATURE_TRANSFORMATIONS.md",
        "## 8. Building Each Model's Transformer",
        "### 8.4",
    ),
    (
        "docs/DIRECT_COHORT_MODEL.md",
        "## 0. The Rebuilt Model",
        "## 1. Statistical Model",
    ),
    (
        "docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md",
        "## 0. The Rebuilt Model",
        "## 1. ",
    ),
]

ns: dict = {"pd": pd, "np": np}
ns["raw_table"] = StudentPopulationSimulator(
    load_simulation_config("configs/simulation.toml")
).run(np.random.default_rng(0))
print(f"raw_table: {ns['raw_table'].shape}")
n = 0
for path, start, stop in SECTIONS:
    for line, code in blocks(path, start, stop):
        n += 1
        exec(compile(code, f"{path}:{line}", "exec"), ns)  # noqa: S102 doc blocks are run on purpose
        if "table" in ns and "train_index" not in ns:
            table = ns["table"]
            ns["train_index"], ns["test_index"] = Splitter(
                "grouped"
            ).train_test_indices(
                table, table["neighborhood_id"], test_size=0.2, random_state=0
            )
            ns["fit_df"] = ns["train_df"] = table.iloc[ns["train_index"]]
            ns["valid_df"] = ns["test_df"] = table.iloc[ns["test_index"]]
        print(f"ok  block {n:2d}  {path}:{line}")
print(f"\n{n} blocks ran.")
for key in ("score", "scores"):
    if key in ns:
        print(key, "=", ns[key])
