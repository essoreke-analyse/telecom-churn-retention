import numpy as np
import pandas as pd

from churn.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from churn.train import cross_validate, make_logistic, recall_at_top


def make_synthetic(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=range(n))
    for c in NUMERIC_FEATURES:
        X[c] = rng.integers(0, 100, n)
    for c in BINARY_FEATURES:
        X[c] = rng.integers(0, 2, n)
    X["age_group"] = rng.choice(["1", "2", "3", "4", "5"], n)
    X["charge_band"] = rng.choice(["0", "1", "2", "3+"], n)
    X = X[ALL_FEATURES]
    y = pd.Series((X["complains"] + rng.random(n) > 1.0).astype(int))
    groups = np.arange(n) // 2  # pairs of identical rows
    return X, y, groups


def test_recall_at_top_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at_top(y, scores, frac=0.2) == 1.0


def test_pipeline_predicts_probabilities():
    X, y, _ = make_synthetic()
    proba = make_logistic().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cv_keeps_groups_together_and_covers_all_rows():
    X, y, groups = make_synthetic()
    folds, oof, fold_id = cross_validate(make_logistic(), X, y, groups, n_splits=5)
    assert len(folds) == 5
    assert (fold_id >= 0).all()
    per_group = pd.Series(fold_id).groupby(groups).nunique()
    assert (per_group == 1).all()  # no group straddles two folds
    assert ((oof >= 0) & (oof <= 1)).all()