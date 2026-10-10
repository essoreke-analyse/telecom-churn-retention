"""Loading, cleaning and validation of the customer churn dataset."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "churn"

EXPECTED_COLUMNS = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
    "churn",
]

# Allowed values for categorical / binary columns (from the data dictionary)
ALLOWED_VALUES = {
    "churn": {0, 1},
    "complains": {0, 1},
    "status": {1, 2},
    "tariff_plan": {1, 2},
}


def load_config(path: Path | str | None = None) -> dict:
    """Read the YAML configuration file."""
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace runs of whitespace by '_'.

    'Call  Failure' (two spaces) -> 'call_failure'
    """
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check that the dataset matches what the rest of the code expects.

    Raises ValueError with an explicit message if a rule is broken.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_null = int(df[EXPECTED_COLUMNS].isnull().sum().sum())
    if n_null > 0:
        raise ValueError(f"Dataset contains {n_null} missing values")

    for col, allowed in ALLOWED_VALUES.items():
        unexpected = set(df[col].unique()) - allowed
        if unexpected:
            raise ValueError(
                f"Column '{col}' has unexpected values {sorted(unexpected)}; "
                f"allowed: {sorted(allowed)}"
            )

    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        logger.warning("%d fully duplicated rows found (kept, to be reviewed in EDA)", n_dup)

    return df


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV, normalize column names and validate it."""
    if path is None:
        path = PROJECT_ROOT / load_config()["paths"]["raw_data"]
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return validate(df)