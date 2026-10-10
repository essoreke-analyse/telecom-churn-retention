"""Model comparison, duplicate-leakage check and ablation study."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from churn.data import load_config, load_raw
from churn.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    duplicate_groups,
    split_xy,
)
from churn.train import cross_validate, score_predictions

MODEL_NAMES = ["logistic", "svm_rbf", "random_forest", "hist_gradient_boosting"]
SUSPECT_FEATURES = ["complains", "is_inactive"]


def _select(cols, wanted):
    return [c for c in wanted if c in cols]


def _prep(cols, log_scale: bool) -> ColumnTransformer:
    """Preprocessing restricted to the columns actually present in X."""
    if log_scale:
        numeric = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
    else:
        numeric = "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, _select(cols, NUMERIC_FEATURES)),
            ("bin", "passthrough", _select(cols, BINARY_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _select(cols, CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0,
    )


def build_model(name: str, cols, seed: int = 42) -> Pipeline:
    cols = list(cols)
    if name == "logistic":
        return Pipeline([("prep", _prep(cols, True)),
                         ("model", LogisticRegression(C=1.0, max_iter=2000))])
    if name == "svm_rbf":
        return Pipeline([("prep", _prep(cols, True)),
                         ("model", SVC(C=1.0, probability=True, random_state=seed))])
    if name == "random_forest":
        return Pipeline([("prep", _prep(cols, False)),
                         ("model", RandomForestClassifier(
                             n_estimators=300, min_samples_leaf=5,
                             class_weight="balanced_subsample",
                             n_jobs=-1, random_state=seed))])
    if name == "hist_gradient_boosting":
        return Pipeline([("prep", _prep(cols, False)),
                         ("model", HistGradientBoostingClassifier(
                             learning_rate=0.05, max_iter=200, max_depth=3,
                             l2_regularization=1.0, random_state=seed))])
    raise ValueError(f"Unknown model: {name}")


def evaluate(name, X, y, groups, active_mask, seed, cv=None) -> dict:
    """Mean CV metrics (+ metrics on active customers from out-of-fold scores)."""
    folds, oof, _ = cross_validate(build_model(name, X.columns, seed), X, y, groups,
                                   seed=seed, cv=cv)
    mean = folds.drop(columns="fold").mean()
    y_np = np.asarray(y)
    active = score_predictions(y_np[active_mask], oof[active_mask])
    return {
        "roc_auc": mean["roc_auc"],
        "pr_auc": mean["pr_auc"],
        "recall_top20": mean["recall_top20"],
        "lift_top20": mean["lift_top20"],
        "active_pr_auc": active["pr_auc"],
        "active_lift_top20": active["lift_top20"],
    }


def main() -> None:
    seed = load_config()["project"]["seed"]
    df = build_features(load_raw())
    X, y = split_xy(df)
    groups = duplicate_groups(df).to_numpy()
    active_mask = (X["is_inactive"] == 0).to_numpy()

    print("=== A. Model comparison (grouped stratified 5-fold CV) ===")
    a = pd.DataFrame({m: evaluate(m, X, y, groups, active_mask, seed)
                      for m in MODEL_NAMES}).T
    print(a.round(3).to_string())

    print("\n=== B. Duplicate leakage: naive split vs grouped split (PR-AUC / ROC-AUC) ===")
    naive = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = {}
    for m in MODEL_NAMES:
        g = a.loc[m]
        n = evaluate(m, X, y, groups, active_mask, seed, cv=naive)
        rows[m] = {"pr_auc_grouped": g["pr_auc"], "pr_auc_naive": n["pr_auc"],
                   "pr_gap": n["pr_auc"] - g["pr_auc"],
                   "roc_grouped": g["roc_auc"], "roc_naive": n["roc_auc"]}
    print(pd.DataFrame(rows).T.round(3).to_string())

    print("\n=== C. Ablation: without complains and is_inactive (grouped CV) ===")
    X_abl = X.drop(columns=SUSPECT_FEATURES)
    c = pd.DataFrame({m: evaluate(m, X_abl, y, groups, active_mask, seed)
                      for m in MODEL_NAMES}).T
    print(c.round(3).to_string())
    print("\nPR-AUC change vs full model:")
    print((c["pr_auc"] - a["pr_auc"]).round(3).to_string())


if __name__ == "__main__":
    main()