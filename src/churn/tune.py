"""Nested cross-validation: hyperparameter tuning (inner) + evaluation (outer)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from churn.compare import SUSPECT_FEATURES, _prep
from churn.data import load_config, load_raw
from churn.features import build_features, duplicate_groups, split_xy
from churn.train import score_predictions

PARAM_GRIDS = {
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [2, 5, 10, 20],
        "model__max_features": ["sqrt", 0.5],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_iter": [100, 200, 400],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 1.0, 5.0],
    },
}


def base_pipeline(name: str, cols, seed: int) -> Pipeline:
    if name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    else:
        model = HistGradientBoostingClassifier(random_state=seed)
    return Pipeline([("prep", _prep(list(cols), False)), ("model", model)])


def nested_cv(name, X, y, groups, seed=42, n_outer=5, n_inner=3, n_iter=12):
    """Outer folds score the tuned model; inner search never sees outer test data."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    oof = np.zeros(len(y))
    rows, chosen = [], []
    for i, (tr, te) in enumerate(outer.split(X, y, groups)):
        search = RandomizedSearchCV(
            base_pipeline(name, X.columns, seed),
            PARAM_GRIDS[name],
            n_iter=n_iter,
            scoring="average_precision",
            cv=inner,
            random_state=seed,
            n_jobs=1,
        )
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups[tr])
        proba = search.predict_proba(X.iloc[te])[:, 1]
        oof[te] = proba
        rows.append({"fold": i, **score_predictions(y.iloc[te], proba)})
        chosen.append(search.best_params_)
    return pd.DataFrame(rows), oof, chosen


def main() -> None:
    seed = load_config()["project"]["seed"]
    df = build_features(load_raw())
    X, y = split_xy(df)
    groups = duplicate_groups(df).to_numpy()
    active = (X["is_inactive"] == 0).to_numpy()
    y_np = y.to_numpy()

    versions = {
        "full": X,
        "early_warning": X.drop(columns=SUSPECT_FEATURES),
    }
    results = {}
    for vname, Xv in versions.items():
        for name in PARAM_GRIDS:
            print(f"Tuning {name} ({vname})...", flush=True)
            folds, oof, chosen = nested_cv(name, Xv, y, groups, seed=seed)
            mean = folds.drop(columns="fold").mean()
            act = score_predictions(y_np[active], oof[active])
            results[(vname, name)] = {
                "roc_auc": mean["roc_auc"],
                "pr_auc": mean["pr_auc"],
                "pr_auc_std": folds["pr_auc"].std(),
                "recall_top20": mean["recall_top20"],
                "lift_top20": mean["lift_top20"],
                "active_pr_auc": act["pr_auc"],
            }
            print("  params chosen per outer fold:")
            for p in chosen:
                print("   ", p)

    print("\n=== Nested CV results (tuned models) ===")
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()