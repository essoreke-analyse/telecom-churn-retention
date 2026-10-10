import numpy as np
import pandas as pd

from churn.features import ALL_FEATURES, BINARY_FEATURES, NUMERIC_FEATURES
from churn.tune import PARAM_GRIDS, nested_cv


def make_synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=range(n))
    for c in NUMERIC_FEATURES:
        X[c] = rng.integers(0, 100, n)
    for c in BINARY_FEATURES:
        X[c] = rng.integers(0, 2, n)
    X["age_group"] = rng.choice(["1", "2", "3"], n)
    X["charge_band"] = rng.choice(["0", "1", "3+"], n)
    X = X[ALL_FEATURES]
    y = pd.Series((X["complains"] + rng.random(n) > 1.0).astype(int))
    return X, y, np.arange(n) // 2


def test_nested_cv_runs_for_both_models():
    X, y, groups = make_synthetic()
    for name in PARAM_GRIDS:
        folds, oof, chosen = nested_cv(
            name, X, y, groups, n_outer=3, n_inner=2, n_iter=2
        )
        assert len(folds) == 3
        assert len(chosen) == 3
        assert ((oof >= 0) & (oof <= 1)).all()