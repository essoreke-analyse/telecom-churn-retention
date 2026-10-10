"""From scores to decisions: expected gain, profit curve and campaign policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.data import PROJECT_ROOT, load_config, load_raw
from churn.features import build_features, duplicate_groups, split_xy
from churn.tune import nested_cv


def expected_gain(proba, value, success_rate: float, cost: float) -> np.ndarray:
    """Expected net gain of contacting a customer, known BEFORE the outcome."""
    return np.asarray(proba) * success_rate * np.asarray(value) - cost


def realized_profit(y, value, success_rate: float, cost: float) -> np.ndarray:
    """Profit per contacted customer once the truth is known.

    A real churner is saved with probability `success_rate` (worth `value`);
    every contacted customer costs `cost`, churner or not.
    """
    return np.asarray(y) * success_rate * np.asarray(value) - cost


def profit_curve(y, value, order, success_rate: float, cost: float) -> np.ndarray:
    """Cumulative profit when contacting the first k customers of `order`.

    Returns an array of length n + 1 (index k = number of customers contacted).
    """
    per_customer = realized_profit(y, value, success_rate, cost)[order]
    return np.concatenate([[0.0], np.cumsum(per_customer)])


def evaluate_policy(y, value, contacted, success_rate: float, cost: float) -> dict:
    """Summarise a campaign defined by a boolean mask of contacted customers."""
    y = np.asarray(y)
    contacted = np.asarray(contacted, dtype=bool)
    profit = realized_profit(y, value, success_rate, cost)[contacted].sum()
    return {
        "contacted": int(contacted.sum()),
        "share_contacted": float(contacted.mean()),
        "churners_reached": int(y[contacted].sum()),
        "recall": float(y[contacted].sum() / y.sum()),
        "profit": float(profit),
    }


def policy_table(y, value, proba, success_rate: float, cost: float) -> pd.DataFrame:
    """Compare the three ways of choosing who to contact."""
    y, value, proba = map(np.asarray, (y, value, proba))
    order = np.argsort(-proba, kind="stable")
    curve = profit_curve(y, value, order, success_rate, cost)
    k_best = int(np.argmax(curve))

    best_mask = np.zeros(len(y), dtype=bool)
    best_mask[order[:k_best]] = True
    gain_mask = expected_gain(proba, value, success_rate, cost) > 0

    rows = {
        "contact everyone": evaluate_policy(y, value, np.ones(len(y), bool), success_rate, cost),
        "top-k by probability (best k, hindsight)": evaluate_policy(y, value, best_mask, success_rate, cost),
        "expected-gain rule (applicable)": evaluate_policy(y, value, gain_mask, success_rate, cost),
    }
    return pd.DataFrame(rows).T


def sensitivity(y, value, proba, success_rates, costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profit and share contacted of the expected-gain rule over an assumptions grid."""
    y, value, proba = map(np.asarray, (y, value, proba))
    profit = pd.DataFrame(index=costs, columns=success_rates, dtype=float)
    share = profit.copy()
    for c in costs:
        for s in success_rates:
            mask = expected_gain(proba, value, s, c) > 0
            res = evaluate_policy(y, value, mask, s, c)
            profit.loc[c, s] = res["profit"]
            share.loc[c, s] = res["share_contacted"]
    profit.index.name = share.index.name = "offer_cost"
    profit.columns.name = share.columns.name = "success_rate"
    return profit, share


def plot_profit_curve(y, value, proba, success_rate, cost, title, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-np.asarray(proba), kind="stable")
    curve = profit_curve(y, value, order, success_rate, cost)
    k_best = int(np.argmax(curve))
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(len(curve)) / len(y) * 100, curve)
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.axvline(k_best / len(y) * 100, color="red", linestyle="--",
                label=f"best: contact top {k_best / len(y):.0%}")
    plt.xlabel("% of customers contacted (highest churn probability first)")
    plt.ylabel("net profit (customer-value units)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    biz = cfg["business"]
    s, c = biz["success_rate"], biz["offer_cost"]

    df = build_features(load_raw())
    X, y = split_xy(df)
    groups = duplicate_groups(df).to_numpy()
    value = X["customer_value"].to_numpy()
    y_np = y.to_numpy()

    versions = {
        "full": X,
        "early_warning": X.drop(columns=["complains", "is_inactive"]),
    }
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    for vname, Xv in versions.items():
        print(f"\n##### Version: {vname} (out-of-fold scores, tuned gradient boosting) #####")
        _, oof, _ = nested_cv("hist_gradient_boosting", Xv, y, groups, seed=seed)

        print(f"\nAssumptions: offer cost = {c}, success rate = {s:.0%}")
        print(policy_table(y_np, value, oof, s, c).round(3).to_string())

        profit, share = sensitivity(y_np, value, oof, biz["sensitivity"]["success_rates"],
                                    biz["sensitivity"]["offer_costs"])
        print("\nProfit of the expected-gain rule (rows: offer cost, columns: success rate)")
        print(profit.round(0).to_string())
        print("\nShare of customers contacted")
        print(share.round(3).to_string())

        everyone = [evaluate_policy(y_np, value, np.ones(len(y_np), bool), sr, oc)["profit"]
                    for oc in biz["sensitivity"]["offer_costs"]
                    for sr in biz["sensitivity"]["success_rates"]]
        print("\nProfit of contacting everyone, same grid (flattened, row by row):")
        print(np.round(everyone, 0))

        plot_profit_curve(y_np, value, oof, s, c,
                          f"Profit curve ({vname} model)",
                          fig_dir / f"profit_curve_{vname}.png")


if __name__ == "__main__":
    main()