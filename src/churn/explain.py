"""Model interpretation: calibration, permutation importance and target profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

from churn.data import PROJECT_ROOT, load_config, load_raw
from churn.evaluate import expected_gain
from churn.features import build_features, duplicate_groups, split_xy
from churn.tune import base_pipeline, nested_cv

# Parameters chosen most often by the nested CV (see tune.py output)
BOOSTING_PARAMS = {
    "model__learning_rate": 0.1,
    "model__max_iter": 400,
    "model__max_depth": 3,
    "model__min_samples_leaf": 10,
    "model__l2_regularization": 1.0,
}


def calibration_table(y, proba, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed churn rate per quantile bin of the score."""
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos}).round(3)


def brier_score(y, proba) -> float:
    return float(np.mean((np.asarray(proba) - np.asarray(y)) ** 2))


def out_of_fold_permutation_importance(X, y, groups, seed=42, n_splits=5, n_repeats=5):
    """Permutation importance computed on held-out folds only (average PR-AUC drop)."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    imps = []
    for tr, te in cv.split(X, y, groups):
        pipe = base_pipeline("hist_gradient_boosting", X.columns, seed)
        pipe.set_params(**BOOSTING_PARAMS).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(
            pipe, X.iloc[te], y.iloc[te], scoring="average_precision",
            n_repeats=n_repeats, random_state=seed, n_jobs=1,
        )
        imps.append(r.importances_mean)
    out = pd.DataFrame({"feature": X.columns, "pr_auc_drop": np.mean(imps, axis=0)})
    return out.sort_values("pr_auc_drop", ascending=False).round(4)


def target_profile(X, y, proba, success_rate, cost) -> pd.DataFrame:
    """Compare customers selected by the expected-gain rule with the others."""
    selected = expected_gain(proba, X["customer_value"], success_rate, cost) > 0
    cols = ["seconds_of_use", "frequency_of_sms", "customer_value", "subscription_length"]
    prof = X.assign(selected=selected, churn=np.asarray(y)).groupby("selected").agg(
        customers=("churn", "size"),
        churn_rate=("churn", "mean"),
        inactive_share=("is_inactive", "mean"),
        complained_share=("complains", "mean"),
        **{f"median_{c}": (c, "median") for c in cols},
    )
    return prof.round(3).T


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    s, c = cfg["business"]["success_rate"], cfg["business"]["offer_cost"]
    fig_dir = PROJECT_ROOT / "reports" / "figures"

    df = build_features(load_raw())
    X, y = split_xy(df)
    groups = duplicate_groups(df).to_numpy()
    y_np = y.to_numpy()

    versions = {"full": X, "early_warning": X.drop(columns=["complains", "is_inactive"])}
    for vname, Xv in versions.items():
        print(f"\n##### Version: {vname} #####")
        _, oof, _ = nested_cv("hist_gradient_boosting", Xv, y, groups, seed=seed)

        print(f"\nBrier score: {brier_score(y_np, oof):.4f} "
              f"(constant prediction = {brier_score(y_np, np.full(len(y_np), y_np.mean())):.4f})")
        print("Calibration (deciles of predicted risk):")
        print(calibration_table(y_np, oof).to_string(index=False))

        print("\nPermutation importance (drop in PR-AUC when the variable is shuffled):")
        print(out_of_fold_permutation_importance(Xv, y, groups, seed=seed).to_string(index=False))

        print(f"\nProfile of customers selected by the expected-gain rule "
              f"(cost={c}, success={s:.0%}):")
        print(target_profile(X, y, oof, s, c).to_string())

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cal = calibration_table(y_np, oof)
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        plt.plot(cal.mean_predicted, cal.observed_rate, "o-", label="model")
        plt.xlabel("predicted churn probability")
        plt.ylabel("observed churn rate")
        plt.title(f"Calibration ({vname} model)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"calibration_{vname}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()