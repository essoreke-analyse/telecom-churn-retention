"""Model pipelines and grouped, stratified cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import load_config, load_raw
from churn.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    duplicate_groups,
    split_xy,
)


def make_linear_preprocessor() -> ColumnTransformer:
    """log1p + standardisation for usage variables, one-hot for categories."""
    numeric = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_logistic(C: float = 1.0) -> Pipeline:
    """Baseline: L2-regularised logistic regression."""
    return Pipeline(
        [
            ("prep", make_linear_preprocessor()),
            ("model", LogisticRegression(C=C, max_iter=2000)),
        ]
    )


def recall_at_top(y_true, scores, frac: float = 0.2) -> float:
    """Share of all churners found among the `frac` highest-scored customers."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = max(1, int(round(frac * len(y_true))))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y_true[top].sum() / y_true.sum())


def score_predictions(y_true, scores, frac: float = 0.2) -> dict:
    """Threshold-free metrics, suited to an imbalanced target."""
    rec = recall_at_top(y_true, scores, frac)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        f"recall_top{int(frac * 100)}": rec,
        f"lift_top{int(frac * 100)}": rec / frac,
    }


def cross_validate(model, X, y, groups, n_splits: int = 5, seed: int = 42, cv=None):
    """Stratified + grouped CV by default (identical rows never straddle folds).

    Pass `cv` to use another splitter (e.g. a naive StratifiedKFold).
    Returns (per-fold metrics, out-of-fold probabilities, fold id per row).
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    if cv is None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(y))
    fold_id = np.full(len(y), -1)
    rows = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        fold_id[test_idx] = i
        rows.append({"fold": i, **score_predictions(y.iloc[test_idx], proba)})
    return pd.DataFrame(rows), oof, fold_id


def main() -> None:
    seed = load_config()["project"]["seed"]
    df = build_features(load_raw())
    X, y = split_xy(df)
    groups = duplicate_groups(df).to_numpy()

    folds, oof, _ = cross_validate(make_logistic(), X, y, groups, seed=seed)

    print("Baseline: logistic regression (5-fold stratified grouped CV)\n")
    print(folds.round(3).to_string(index=False))
    print("\nMean +/- std across folds")
    summary = folds.drop(columns="fold").agg(["mean", "std"]).T.round(3)
    print(summary)

    print(f"\nRandom-model reference: PR-AUC = churn rate = {y.mean():.3f}, lift = 1.0")

    active = (X["is_inactive"] == 0).to_numpy()
    y_np = y.to_numpy()
    print(f"\nActive customers only (n={active.sum()}, churners={y_np[active].sum()}):")
    for k, v in score_predictions(y_np[active], oof[active]).items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()