"""PR #12 validation V4: the plan's §6 end-to-end probe, verbatim (two estimators, one with the offset)."""

import warnings

warnings.simplefilter("error")
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import DirectCohortModel, IndependentCohortModels
from age_group_prediction.scoring import POISSON_DEVIANCE

rng = np.random.default_rng(0)
n = 400
table = pd.DataFrame(
    {"x": rng.normal(size=n), "n_apartments": rng.integers(5, 40, n).astype(float)}
)
table["n_kindergarten"] = rng.poisson(
    0.1 * table["n_apartments"] * np.exp(0.5 * table["x"])
)
table["n_elementary"] = rng.poisson(np.exp(1 - 0.5 * table["x"]))
tree = FeatureTransformer(
    (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
)
y = table[["n_kindergarten", "n_elementary"]]
exposure = table["n_apartments"].to_numpy()

model = IndependentCohortModels(
    {
        "n_kindergarten": DirectCohortModel(
            estimator=LGBMRegressor(objective="poisson", n_jobs=1),
            use_exposure=True,
            feature_transformer=tree,
        ),  # LightGBM
        "n_elementary": DirectCohortModel(
            estimator=HistGradientBoostingRegressor(loss="poisson"),
            feature_transformer=tree,
        ),  # sklearn, no offset
    }
).fit(table, y, exposure=exposure)
pred = model.predict(table, exposure=exposure)
assert list(pred.columns) == list(y.columns) and pred.index.equals(table.index)
print({c: round(model.evaluate(y[c], pred[c], POISSON_DEVIANCE), 3) for c in y})
