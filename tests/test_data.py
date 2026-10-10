import pandas as pd
import pytest

from churn.data import (
    EXPECTED_COLUMNS,
    PROJECT_ROOT,
    load_config,
    load_raw,
    normalize_columns,
    validate,
)


def make_raw_df() -> pd.DataFrame:
    """Tiny dataframe with the ORIGINAL column names (double spaces included)."""
    return pd.DataFrame(
        {
            "Call  Failure": [0, 5, 10, 2],
            "Complains": [0, 0, 1, 0],
            "Subscription  Length": [30, 12, 40, 25],
            "Charge  Amount": [0, 1, 0, 3],
            "Seconds of Use": [1000, 200, 0, 5000],
            "Frequency of use": [40, 5, 0, 90],
            "Frequency of SMS": [10, 0, 0, 100],
            "Distinct Called Numbers": [10, 2, 0, 30],
            "Age Group": [3, 2, 3, 4],
            "Tariff Plan": [1, 1, 1, 2],
            "Status": [1, 2, 2, 1],
            "Age": [30, 25, 32, 40],
            "Customer Value": [300.5, 50.0, 0.0, 900.2],
            "Churn": [0, 1, 1, 0],
        }
    )


def test_normalize_columns_handles_double_spaces():
    df = normalize_columns(make_raw_df())
    assert list(df.columns) == EXPECTED_COLUMNS


def test_validate_accepts_clean_data():
    df = normalize_columns(make_raw_df())
    assert validate(df) is df


def test_validate_rejects_missing_column():
    df = normalize_columns(make_raw_df()).drop(columns=["churn"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate(df)


def test_validate_rejects_missing_values():
    df = normalize_columns(make_raw_df())
    df.loc[0, "age"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate(df)


def test_validate_rejects_invalid_target():
    df = normalize_columns(make_raw_df())
    df.loc[0, "churn"] = 2
    with pytest.raises(ValueError, match="churn"):
        validate(df)


def test_load_raw_from_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_raw_df().to_csv(csv_path, index=False)
    df = load_raw(csv_path)
    assert df.shape == (4, 14)
    assert "call_failure" in df.columns


REAL_DATA = PROJECT_ROOT / load_config()["paths"]["raw_data"]


@pytest.mark.skipif(not REAL_DATA.exists(), reason="raw dataset not available locally")
def test_real_dataset_matches_documentation():
    df = load_raw()
    assert len(df) == 3150
    assert df["churn"].sum() == 495  # 15.71 % of the sample