import numpy as np
import pandas as pd
import pytest

from churn.compare import MODEL_NAMES, SUSPECT_FEATURES, build_model
from churn.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    NUMERIC_FEATURES,
)


def make_synthetic(n: int = 200, seed: int = 0):
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
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name):
    X, y = make_synthetic()
    proba = build_model(name, X.columns).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_work_after_ablation(name):
    X, y = make_synthetic()
    X = X.drop(columns=SUSPECT_FEATURES)
    proba = build_model(name, X.columns).fit(X, y).predict_proba(X)
    assert proba.shape[0] == len(X)