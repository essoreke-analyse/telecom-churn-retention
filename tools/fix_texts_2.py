"""Apply the notebook 03 text corrections to tools/make_notebooks.py (run once)."""
from pathlib import Path

path = Path(__file__).resolve().parent / "make_notebooks.py"
text = path.read_text(encoding="utf-8")

REPLACEMENTS = [
    # 1. legend hiding the best bar
    (
        'ax.set_title("Model comparison")\nax.legend()',
        'ax.set_title("Model comparison")\nax.legend(loc="lower right")',
    ),
    # 2. ablation reading
    (
        "**Reading.** The logistic regression collapses (it leaned heavily on these two variables), while the tree-based models keep most of their performance: usage carries most of the signal. This justifies",
        "**Reading.** The logistic regression collapses (PR-AUC -0.215: it leaned heavily on these two variables) and the SVM loses 0.10, while the tree-based models lose 0.065 to 0.080 and keep most of their performance (PR-AUC about 0.84 before tuning). Usage therefore carries a large part of the signal, but the two variables still add something. This justifies",
    ),
    # 3. decision in section D
    (
        "It has the best PR-AUC in both, it is more stable in the early-warning setting (fold standard deviation 0.039 against 0.074 for the random forest), and it is better on still-active customers, the real target of a campaign. In the full version the two models are tied within noise.",
        "It has the best PR-AUC in both and is more stable in the early-warning setting (fold standard deviation 0.039 against 0.074 for the random forest). On still-active customers, the real target of a campaign, it is clearly better in the early-warning version (PR-AUC 0.854 against 0.811, from `tune.py`) and tied in the full version (0.926 against 0.932). In the full version the two models are tied within noise, so the choice there is one of consistency.",
    ),
    # 4a. source of the figures
    (
        "Computed on out-of-fold scores of the tuned model:",
        "Computed on out-of-fold scores of the tuned gradient boosting (from `python -m churn.evaluate`, seed 42):",
    ),
    # 4b. row label
    (
        "| **Expected-gain rule** (applicable) |",
        "| **Expected-gain rule, full model** (applicable) |",
    ),
    # 4c. note under the table
    (
        "| Expected-gain rule, early-warning model | 13% | 63% | +12,004 |",
        "| Expected-gain rule, early-warning model | 13% | 63% | +12,004 |\n\nThe expected-gain rule beats ranking by probability alone because it also uses each customer's value. The hindsight row picks k using the true outcomes, so it is a reference, not a deployable rule.",
    ),
    # 5. conclusions
    (
        "reaches a PR-AUC of about 0.95 (random: 0.16)",
        "reaches a PR-AUC of about 0.95 in the full version (random: 0.16)",
    ),
    (
        "3. Without the two possibly-late variables the model keeps a PR-AUC of about 0.91: it genuinely detects departures from usage.",
        "3. Without the two possibly-late variables, the tuned gradient boosting keeps a PR-AUC of about 0.91 (untuned models: about 0.84), so most of the signal comes from usage.",
    ),
    (
        "The result holds across the cost and success-rate grid tested (see `evaluate.py`), although the assumptions themselves are invented.",
        "The profit stays positive across the whole cost and success-rate grid tested (see `evaluate.py`), but in the worst case (10% success, cost 20) it is small (+863, with only 2% of customers contacted), and the assumptions themselves are invented.",
    ),
]

for old, new in REPLACEMENTS:
    assert old in text, f"Text not found, nothing written:\n{old[:80]}..."
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print(f"{len(REPLACEMENTS)} corrections applied to {path.name}")