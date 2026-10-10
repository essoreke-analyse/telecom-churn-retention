import numpy as np
import pandas as pd
import pytest

from churn.data import normalize_columns
from churn.features import build_features, split_xy
from churn.final import SUSPECT_FEATURES, fit_final, load_bundle, save_bundle
from churn.predict import score_customers


def make_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the ORIGINAL raw schema (double spaces included)."""
    rng = np.random.default_rng(seed)
    age_group = rng.integers(1, 6, n)
    status = rng.choice([1, 2], n, p=[0.75, 0.25])
    seconds = rng.integers(0, 10000, n)
    churn = ((status == 2) & (rng.random(n) < 0.8)) | ((seconds < 500) & (rng.random(n) < 0.5))
    return pd.DataFrame(
        {
            "Call  Failure": rng.integers(0, 30, n),
            "Complains": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "Subscription  Length": rng.integers(3, 47, n),
            "Charge  Amount": rng.integers(0, 11, n),
            "Seconds of Use": seconds,
            "Frequency of use": seconds // 60,
            "Frequency of SMS": rng.integers(0, 200, n),
            "Distinct Called Numbers": rng.integers(0, 60, n),
            "Age Group": age_group,
            "Tariff Plan": rng.choice([1, 2], n, p=[0.9, 0.1]),
            "Status": status,
            "Age": np.array([15, 25, 30, 45, 55])[age_group - 1],
            "Customer Value": rng.uniform(0, 1500, n),
            "Churn": churn.astype(int),
        }
    )


def make_bundle(raw, version="full"):
    X, y = split_xy(build_features(normalize_columns(raw)))
    return fit_final(X, y, version)


def test_score_customers_output_shape_and_order():
    raw = make_raw()
    scored = score_customers(raw.drop(columns="Churn"), make_bundle(raw), 0.3, 10)
    assert len(scored) == len(raw)
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["expected_gain"].is_monotonic_decreasing


def test_contact_flag_matches_gain_rule():
    raw = make_raw()
    scored = score_customers(raw, make_bundle(raw), 0.3, 10)
    assert (scored["contact"] == (scored["expected_gain"] > 0)).all()


def test_early_warning_does_not_use_suspect_features():
    bundle = make_bundle(make_raw(), "early_warning")
    assert not set(SUSPECT_FEATURES) & set(bundle["features"])


def test_save_and_load_roundtrip(tmp_path):
    raw = make_raw()
    bundle = make_bundle(raw)
    save_bundle(bundle, tmp_path / "churn_full.joblib")
    loaded = load_bundle("full", model_dir=tmp_path)
    a = score_customers(raw, bundle, 0.3, 10)
    b = score_customers(raw, loaded, 0.3, 10)
    assert np.allclose(a["churn_probability"], b["churn_probability"])


def test_missing_column_raises_clear_error():
    raw = make_raw()
    bundle = make_bundle(raw)
    with pytest.raises(ValueError, match="Missing columns"):
        score_customers(raw.drop(columns=["Seconds of Use"]), bundle, 0.3, 10)