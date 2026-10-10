"""Score customers and list who to contact (expected-gain rule)."""
from __future__ import annotations

import argparse

import pandas as pd

from churn.data import PROJECT_ROOT, load_config, normalize_columns, validate
from churn.evaluate import expected_gain
from churn.features import ALL_FEATURES, build_features
from churn.final import VERSIONS, load_bundle


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw customer table (original column names) -> modelling features.

    The raw schema must contain the 13 explanatory columns. `churn` is optional
    (unknown for new customers).
    """
    df = normalize_columns(raw)
    if "churn" not in df.columns:
        df["churn"] = 0  # placeholder, never used for scoring
    df = validate(df)
    return build_features(df)[ALL_FEATURES]


def score_customers(raw: pd.DataFrame, bundle: dict, success_rate: float, cost: float) -> pd.DataFrame:
    """Churn probability, expected gain and contact decision for each customer."""
    feats = prepare_features(raw)
    proba = bundle["pipeline"].predict_proba(feats[bundle["features"]])[:, 1]
    gain = expected_gain(proba, feats["customer_value"], success_rate, cost)
    out = pd.DataFrame(
        {
            "customer_value": feats["customer_value"].to_numpy(),
            "churn_probability": proba,
            "expected_gain": gain,
            "contact": gain > 0,
        },
        index=raw.index,
    )
    return out.sort_values("expected_gain", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for a retention campaign.")
    parser.add_argument("--input", help="CSV with the raw schema (default: the training CSV, demo only)")
    parser.add_argument("--version", choices=VERSIONS, default="full")
    parser.add_argument("--output", default="data/processed/scored_customers.csv")
    args = parser.parse_args()

    cfg = load_config()
    biz = cfg["business"]
    src = args.input or (PROJECT_ROOT / cfg["paths"]["raw_data"])
    raw = pd.read_csv(src)

    scored = score_customers(raw, load_bundle(args.version), biz["success_rate"], biz["offer_cost"])
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index_label="customer_row")

    n = int(scored["contact"].sum())
    print(f"Model: {args.version} | customers scored: {len(scored)}")
    print(f"To contact: {n} ({n / len(scored):.1%}) | "
          f"offer cost={biz['offer_cost']}, success rate={biz['success_rate']:.0%}")
    print("\nTop 10 by expected gain:")
    print(scored.head(10).round(3).to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()