"""Deterministic feature engineering (no learning from data, so no leakage)."""
from __future__ import annotations

import pandas as pd

TARGET = "churn"

NUMERIC_FEATURES = [
    "call_failure",
    "subscription_length",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "customer_value",
]
BINARY_FEATURES = ["complains", "is_inactive", "is_contract"]
CATEGORICAL_FEATURES = ["age_group", "charge_band"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the validated raw dataframe into the modelling dataframe.

    - drops `age` (one-to-one recoding of `age_group`)
    - recodes `status` and `tariff_plan` (1/2) as clear 0/1 flags
    - groups `charge_amount` into 0 / 1 / 2 / 3+ (sparse upper levels)
    - keeps `age_group` as a category (its effect is not monotonic)
    """
    out = df.copy()

    out["is_inactive"] = (out["status"] == 2).astype(int)
    out["is_contract"] = (out["tariff_plan"] == 2).astype(int)
    out["charge_band"] = out["charge_amount"].clip(upper=3).astype(int).astype(str)
    out.loc[out["charge_band"] == "3", "charge_band"] = "3+"
    out["age_group"] = out["age_group"].astype(str)

    out = out.drop(columns=["age", "status", "tariff_plan", "charge_amount"])
    return out[ALL_FEATURES + [TARGET]]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""
    return df[ALL_FEATURES], df[TARGET]


def duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Group id for rows with identical features (target excluded).

    Used so that identical rows stay on the same side of a train/test split.
    """
    cols = [c for c in df.columns if c != TARGET]
    return df.groupby(cols, sort=False, observed=True).ngroup()