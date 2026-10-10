import pandas as pd

from churn.features import (
    ALL_FEATURES,
    TARGET,
    build_features,
    duplicate_groups,
    split_xy,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_failure": [0, 5, 10, 2],
            "complains": [0, 0, 1, 0],
            "subscription_length": [30, 12, 40, 25],
            "charge_amount": [0, 1, 7, 3],
            "seconds_of_use": [1000, 200, 0, 5000],
            "frequency_of_use": [40, 5, 0, 90],
            "frequency_of_sms": [10, 0, 0, 100],
            "distinct_called_numbers": [10, 2, 0, 30],
            "age_group": [3, 2, 3, 4],
            "tariff_plan": [1, 1, 1, 2],
            "status": [1, 2, 2, 1],
            "age": [30, 25, 30, 45],
            "customer_value": [300.5, 50.0, 0.0, 900.2],
            "churn": [0, 1, 1, 0],
        }
    )


def test_build_features_drops_and_recodes():
    out = build_features(make_df())
    assert "age" not in out.columns
    assert "status" not in out.columns
    assert list(out["is_inactive"]) == [0, 1, 1, 0]
    assert list(out["is_contract"]) == [0, 0, 0, 1]


def test_charge_band_groups_high_levels():
    out = build_features(make_df())
    assert list(out["charge_band"]) == ["0", "1", "3+", "3+"]


def test_age_group_is_categorical_string():
    out = build_features(make_df())
    assert out["age_group"].tolist() == ["3", "2", "3", "4"]


def test_output_columns():
    out = build_features(make_df())
    assert list(out.columns) == ALL_FEATURES + [TARGET]


def test_split_xy_has_no_target():
    X, y = split_xy(build_features(make_df()))
    assert TARGET not in X.columns
    assert list(y) == [0, 1, 1, 0]


def test_duplicate_groups_ignore_target():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]].assign(churn=1)], ignore_index=True)
    groups = duplicate_groups(build_features(df))
    assert groups.iloc[0] == groups.iloc[4]
    assert groups.nunique() == 4


def test_real_dataset_features():
    from churn.data import PROJECT_ROOT, load_config, load_raw

    path = PROJECT_ROOT / load_config()["paths"]["raw_data"]
    if not path.exists():
        return
    out = build_features(load_raw())
    assert out.shape == (3150, len(ALL_FEATURES) + 1)
    assert out.isnull().sum().sum() == 0
    assert duplicate_groups(out).nunique() < 3150