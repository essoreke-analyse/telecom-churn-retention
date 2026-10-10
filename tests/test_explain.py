import numpy as np
import pandas as pd

from churn.explain import brier_score, calibration_table, target_profile


def test_brier_score_perfect_and_worst():
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [1.0, 0.0]) == 1.0


def test_calibration_table_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    cal = calibration_table(y, p, n_bins=5)
    assert len(cal) == 5
    assert np.allclose(cal.mean_predicted, cal.observed_rate, atol=0.06)


def test_target_profile_splits_selected_customers():
    X = pd.DataFrame({
        "customer_value": [200.0, 50.0, 300.0, 10.0],
        "is_inactive": [1, 0, 1, 0],
        "complains": [1, 0, 0, 0],
        "seconds_of_use": [100, 2000, 50, 3000],
        "frequency_of_sms": [0, 20, 5, 40],
        "subscription_length": [30, 30, 20, 40],
    })
    y = [1, 0, 1, 0]
    proba = np.array([0.9, 0.1, 0.8, 0.05])
    prof = target_profile(X, y, proba, success_rate=0.3, cost=10)
    assert prof.loc["customers", True] == 2
    assert prof.loc["churn_rate", True] == 1.0