import numpy as np

from churn.evaluate import (
    evaluate_policy,
    expected_gain,
    policy_table,
    profit_curve,
    sensitivity,
)


def test_expected_gain_matches_worked_example():
    # clients A, B, C from the explanation: p, value
    gain = expected_gain([0.8, 0.4, 0.1], [200, 100, 50], success_rate=0.3, cost=10)
    assert np.allclose(gain, [38.0, 2.0, -8.5])


def test_profit_curve_simple_case():
    y = np.array([1, 0])
    value = np.array([100, 100])
    curve = profit_curve(y, value, order=np.array([0, 1]), success_rate=0.5, cost=10)
    assert np.allclose(curve, [0.0, 40.0, 30.0])


def test_evaluate_policy_counts():
    y = np.array([1, 1, 0, 0])
    value = np.array([100, 100, 100, 100])
    res = evaluate_policy(y, value, [True, False, True, False], 0.5, 10)
    assert res["contacted"] == 2
    assert res["churners_reached"] == 1
    assert res["recall"] == 0.5
    assert np.isclose(res["profit"], 40.0 - 10.0)


def test_hindsight_top_k_beats_contacting_everyone():
    rng = np.random.default_rng(0)
    n = 500
    proba = rng.random(n)
    y = (rng.random(n) < proba).astype(int)
    value = rng.uniform(0, 300, n)
    table = policy_table(y, value, proba, success_rate=0.3, cost=10)
    best = table.loc["top-k by probability (k chosen with hindsight)", "profit"]
    everyone = table.loc["contact everyone", "profit"]
    assert best >= everyone  # k = n is one of the candidates


def test_sensitivity_grid_shape_and_higher_success_not_worse():
    rng = np.random.default_rng(1)
    n = 400
    proba = rng.random(n)
    y = (rng.random(n) < proba).astype(int)
    value = rng.uniform(50, 300, n)
    profit, share = sensitivity(y, value, proba, [0.1, 0.3, 0.5], [5, 20])
    assert profit.shape == (2, 3)
    assert (share.values >= 0).all() and (share.values <= 1).all()
    assert profit.loc[5, 0.5] >= profit.loc[5, 0.1]