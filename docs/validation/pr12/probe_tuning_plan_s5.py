"""PR #12 validation V4: HYPERPARAMETER_TUNING_PLAN.md §5 runs as written, with a seeded-Optuna
stand-in for HyperparameterStudy (Phase 3), on a simulated table."""

import re
import warnings
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMRegressor  # noqa: F401

from age_group_prediction.feature_engineering import (
    ColumnPlan,
    FeatureTransformer,
    OneHot,
)

# Names the doc block uses; it runs through exec, so ruff sees them as unused.
from age_group_prediction.hyperparameter_tuning import (  # noqa: F401
    CVHyperparameterEvaluator,
    FloatParameter,
    IntParameter,
)
from age_group_prediction.modeling import DirectCohortModel  # noqa: F401
from age_group_prediction.preprocessing import (  # noqa: F401
    ExposureTransformer,
    ShareTransformer,
)
from age_group_prediction.scoring import POISSON_DEVIANCE  # noqa: F401
from age_group_prediction.splitting import Splitter  # noqa: F401
from age_group_prediction.utils import take_rows  # noqa: F401
from student_simulator.config import load_simulation_config
from student_simulator.pipeline import StudentPopulationSimulator

warnings.simplefilter("error")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterStudy:
    """Stand-in: a seeded TPE study; optimize returns the study, so .best_params works."""

    def __init__(self, *, seed: int, n_trials: int) -> None:
        self.seed, self.n_trials = seed, n_trials

    def optimize(self, objective):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study


text = Path("docs/HYPERPARAMETER_TUNING_PLAN.md").read_text()
section = text[text.index("## 5. Worked example") : text.index("## 6.")]
code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
code = code.replace(
    "n_trials=30", "n_trials=3"
)  # time only; the block is otherwise verbatim

raw = StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(
    np.random.default_rng(0)
)
df = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(raw)
groups = df["neighborhood_id"].to_numpy()
cohort_columns = ["n_kindergarten", "n_elementary", "n_highschool"]
seeds = {c: i for i, c in enumerate(cohort_columns)}
models = {}
result = None  # bound by the doc block, one study per cohort
tree = FeatureTransformer(
    plans=(
        ColumnPlan(
            name="numeric",
            columns=(
                "4_rooms_share",
                "5_rooms_share",
                "6_rooms_share",
                "ses",
                "avg_household_size",
                "median_age",
                "n_daycares_500m",
                "n_apartments",
            ),
        ),
        ColumnPlan(
            name="school",
            columns="school_status",
            transforms=(
                OneHot(
                    categories=("none", "existing", "planned"),
                    reference_category="none",
                ),
            ),
        ),
    )
)
exec(compile(code, "HYPERPARAMETER_TUNING_PLAN.md §5", "exec"), globals())  # noqa: S102 doc blocks are run on purpose
for cohort, model in models.items():
    assert model.feature_transformer_ is not None and model.use_exposure_
    print(
        cohort,
        "best:",
        {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in result.best_params.items()
        }
        if cohort == cohort_columns[-1]
        else "fitted",
        "trees:",
        model.estimator_.n_estimators,
    )
print("§5 block ran with n_trials=3 per cohort.")
