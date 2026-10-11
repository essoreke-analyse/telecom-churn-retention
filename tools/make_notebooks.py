"""Generate notebooks 02 and 03 (narrative + code). Run once, then execute the notebooks."""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, NB_DIR / name)
    print("written", NB_DIR / name)


# ----------------------------------------------------------------------------
# Notebook 02: feature analysis
# ----------------------------------------------------------------------------
nb02 = [
    md(r"""
# 02. Feature analysis: what really drives churn?

**Question.** Which variables carry the signal, alone and combined, and which ones should the business watch?

**Why it matters.** A variable can look useless on its own and still be decisive once others are controlled. Conversely, a variable can look decisive and simply be a *consequence* of leaving. Both cases change what the operator should do.

Importance below is measured **out-of-fold** (on customers the model did not train on), with grouped CV so identical rows never sit in both train and test.
"""),
    code(r"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from churn.data import load_raw
from churn.explain import out_of_fold_permutation_importance
from churn.features import (
    ALL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    build_features, duplicate_groups, split_xy,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

raw = load_raw()
df = build_features(raw)
X, y = split_xy(df)
groups = duplicate_groups(df).to_numpy()
print(f"raw: {raw.shape} -> modelling table: {df.shape}")
print(f"churn rate: {y.mean():.1%}")
"""),
    md(r"""
## 1. From 14 columns to 12 features

Decisions taken during the EDA, applied in `src/churn/features.py` (deterministic, so no leakage):

- `age` dropped: only 5 values, a one-to-one recoding of `age_group`.
- `status` and `tariff_plan` (coded 1/2) become clear 0/1 flags `is_inactive` and `is_contract`.
- `charge_amount` (0 to 10, sparse at the top) grouped into `0 / 1 / 2 / 3+`.
- `age_group` kept as a category: its effect is not monotonic.
"""),
    code(r"""
pd.DataFrame(
    {
        "numeric (log + scaling for linear models)": pd.Series(NUMERIC_FEATURES),
        "binary": pd.Series(BINARY_FEATURES),
        "categorical (one-hot)": pd.Series(CATEGORICAL_FEATURES),
    }
).fillna("")
"""),
    md(r"""
## 2. Alone versus together

Two ways to ask "does this variable matter?":

- **Alone** (*univariate power*): how well does the variable separate churners from non-churners by itself? 0.5 = no separation, 1.0 = perfect. Computed with Mann-Whitney.
- **Together** (*permutation importance*): how much does the tuned gradient boosting lose in PR-AUC when this variable is shuffled?

Two versions of the model are shown. The **early-warning** version drops `complains` and `is_inactive`, which may be consequences of leaving (the dataset does not say when they were measured).

*Univariate power is computed for numeric variables only (Mann-Whitney). For binary and categorical variables, the association with churn is shown in notebook 01 (Cramér's V). That is why some cells are empty.*
"""),
    code(r"""
uni = {}
for c in NUMERIC_FEATURES:
    u, _ = mannwhitneyu(X.loc[y == 1, c], X.loc[y == 0, c])
    auc = u / ((y == 1).sum() * (y == 0).sum())
    uni[c] = max(auc, 1 - auc)
uni = pd.Series(uni, name="univariate_power")

imp_full = out_of_fold_permutation_importance(X, y, groups).set_index("feature")["pr_auc_drop"]
imp_early = out_of_fold_permutation_importance(
    X.drop(columns=["complains", "is_inactive"]), y, groups
).set_index("feature")["pr_auc_drop"]

table = pd.concat(
    [uni, imp_full.rename("importance_full"), imp_early.rename("importance_early_warning")], axis=1
).sort_values("importance_full", ascending=False)
table.round(3)
"""),
    code(r"""
plot_df = table[["importance_full", "importance_early_warning"]].sort_values("importance_full")
plot_df.plot.barh(figsize=(8, 6))
plt.xlabel("drop in PR-AUC when the variable is shuffled")
plt.title("Permutation importance (out-of-fold)")
plt.tight_layout()
plt.savefig("../reports/figures/permutation_importance.png", dpi=150)
plt.show()
"""),
    md(r"""
**How to read this.**

- **Usage dominates.** `frequency_of_use` is the top variable in both versions, and `seconds_of_use` and `customer_value` are the best single separators. Customers who use the service less are the ones who leave. Because these usage variables are strongly correlated, the ranking among them should not be over-interpreted.
- **`call_failure` and `subscription_length` are the surprise.** Alone they barely separate churners (univariate power close to 0.5), yet they weigh heavily once usage is controlled. `call_failure` is correlated with usage (about 0.5), so its effect is probably only visible *conditionally*; this is a plausible reading, not something these data prove. Dropping them because of a univariate test, as the academic version suggested, would have thrown away real signal.
- **`complains` matters, `is_inactive` less than expected.** Part of the inactivity signal is already contained in usage.
- **`charge_band` is useless in the full model but useful in the early-warning one** (0.048), where it partly substitutes for the removed variables.
- **Caution.** Usage variables are strongly correlated (up to 0.94), so permutation importance is shared between them: read values as orders of magnitude. Importance means "helps predict", not "causes churn".
"""),
    md(r"""
## 3. Who churns: complaints and inactivity combined
"""),
    code(r"""
seg = df.assign(churn=y).pivot_table(
    index="complains", columns="is_inactive", values="churn", aggfunc=["mean", "size", "sum"]
)
seg.columns = [f"{a}_{'inactive' if b else 'active'}" for a, b in seg.columns]
seg.rename(columns={"mean_active": "churn_rate_active", "mean_inactive": "churn_rate_inactive",
                    "size_active": "n_active", "size_inactive": "n_inactive",
                    "sum_active": "churners_active", "sum_inactive": "churners_inactive"}).round(3)
"""),
    md(r"""
Active customers who did not complain are the hardest to spot: they still contain 66 churners among 2,285 customers (2.9%), more than the 59 churners among active complainers. A rule such as "contact those who complain" would miss most departures, which is why the model must rely on usage.
"""),
    md(r"""
## 4. The business angle: how much value leaves?
"""),
    code(r"""
value = raw.groupby("churn")["customer_value"].agg(["count", "sum", "mean", "median"]).round(1)
value["share_of_total_value"] = (value["sum"] / value["sum"].sum()).round(3)
value
"""),
    md(r"""
## Conclusions

1. Usage is the strongest behavioural signal, and it is observable without any complaint.
2. Variables that look weak alone (`call_failure`, `subscription_length`) matter in combination: never filter variables on univariate tests only.
3. `complains` and `is_inactive` are strong but possibly late signals: hence two model versions.
4. Churners are a small share of total customer value, so retention offers must stay cheap. This is what the decision layer (notebook 03 and `evaluate.py`) quantifies.

Association is not causation: recommendations such as improving network quality need A/B tests.
"""),
]

# ----------------------------------------------------------------------------
# Notebook 03: model comparison
# ----------------------------------------------------------------------------
nb03 = [
    md(r"""
# 03. Model comparison: which model, and can we trust the scores?

**Question.** Does a more complex model beat a simple baseline, are the scores inflated by duplicates, and what is left without the suspicious variables?

**Protocol.** Every model is scored on **exactly the same folds**: stratified (same churn share in each fold) and grouped (identical rows never straddle train and test). Metrics are threshold-free and suited to an imbalanced target (15.7% churners):

- **PR-AUC**: a random model scores 0.157.
- **Recall in the top 20%**: share of all churners found by contacting the 20% highest-scored customers.
- **Lift in the top 20%**: that recall divided by 0.2 (random = 1).

Accuracy is not used: "nobody churns" already scores 84.3%.

*The three experiments below take a few minutes to run.*
"""),
    code(r"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from sklearn.model_selection import StratifiedKFold

from churn.compare import MODEL_NAMES, SUSPECT_FEATURES, evaluate
from churn.data import load_config, load_raw
from churn.features import build_features, duplicate_groups, split_xy

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

seed = load_config()["project"]["seed"]
df = build_features(load_raw())
X, y = split_xy(df)
groups = duplicate_groups(df).to_numpy()
active_mask = (X["is_inactive"] == 0).to_numpy()
print(X.shape, f"churn rate {y.mean():.3f}")
"""),
    md(r"""
## A. Four models, same folds

Hyperparameters here are reasonable hand-picked values (tuning comes later).
"""),
    code(r"""
results = pd.DataFrame({m: evaluate(m, X, y, groups, active_mask, seed) for m in MODEL_NAMES}).T
results.round(3)
"""),
    code(r"""
ax = results["pr_auc"].sort_values().plot.barh(figsize=(7, 3.5), color="#4C72B0")
ax.axvline(y.mean(), color="red", linestyle="--", label=f"random model ({y.mean():.3f})")
ax.set_xlabel("PR-AUC (grouped stratified 5-fold CV)")
ax.set_title("Model comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../reports/figures/model_comparison.png", dpi=150)
plt.show()
"""),
    md(r"""
**Reading.** The tree-based models beat the logistic regression by far more than the fold-to-fold noise (about 0.03 PR-AUC). There are non-linearities and interactions a linear model cannot capture. Random forest and gradient boosting are statistically tied here. The SVM is dominated and was dropped from later steps.
"""),
    md(r"""
## B. Were the scores inflated by duplicate rows?

The dataset contains 300 exact duplicate rows (9.5%). With a naive split, a customer can be in train while his twin is in test, which would flatter flexible models. We compare the grouped split with a naive one.
"""),
    code(r"""
naive = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
rows = {}
for m in MODEL_NAMES:
    n = evaluate(m, X, y, groups, active_mask, seed, cv=naive)
    rows[m] = {"pr_auc_grouped": results.loc[m, "pr_auc"], "pr_auc_naive": n["pr_auc"]}
leak = pd.DataFrame(rows).T
leak["gap"] = leak["pr_auc_naive"] - leak["pr_auc_grouped"]
leak.round(3)
"""),
    md(r"""
**Reading.** The gap is within a few thousandths for every model, so duplicates did **not** inflate the scores. Grouped splitting is kept anyway: it costs nothing and removes the risk by design.
"""),
    md(r"""
## C. Ablation: what if `complains` and `is_inactive` are consequences of leaving?

The dataset does not say when these two variables were measured. If they were recorded late, a model using them partly "recognises customers who already left". We remove them and look at what survives.
"""),
    code(r"""
X_abl = X.drop(columns=SUSPECT_FEATURES)
abl = pd.DataFrame({m: evaluate(m, X_abl, y, groups, active_mask, seed) for m in MODEL_NAMES}).T
print("Without complains and is_inactive:")
display(abl.round(3))
print("PR-AUC change versus the full model:")
display((abl["pr_auc"] - results["pr_auc"]).round(3).to_frame("pr_auc_change"))
"""),
    md(r"""
**Reading.** The logistic regression collapses (PR-AUC -0.215: it leaned heavily on these two variables) and the SVM loses 0.10, while the tree-based models lose 0.065 to 0.080 and keep most of their performance (PR-AUC about 0.84 before tuning). Usage therefore carries a large part of the signal, but the two variables still add something. This justifies delivering **two versions** of the model: a *full* one, and an *early-warning* one that stays valid in the worst case.
"""),
    md(r"""
## D. Tuning with nested cross-validation

Tuning on the same folds that score the model gives optimistic numbers. We use **nested CV**: an inner loop picks hyperparameters on training data only, an outer loop scores the tuned model on unseen folds.

These values come from `python -m churn.tune` (several minutes to run, so reported rather than recomputed here):
"""),
    code(r"""
tuned = pd.DataFrame(
    {
        "roc_auc": [0.985, 0.987, 0.977, 0.978],
        "pr_auc": [0.932, 0.947, 0.867, 0.906],
        "pr_auc_std_across_folds": [0.031, 0.028, 0.074, 0.039],
        "recall_top20": [0.953, 0.951, 0.949, 0.927],
    },
    index=pd.MultiIndex.from_tuples(
        [("full", "random_forest"), ("full", "gradient_boosting"),
         ("early_warning", "random_forest"), ("early_warning", "gradient_boosting")]
    ),
)
tuned
"""),
    md(r"""
**Decision: gradient boosting for both versions.** It has the best PR-AUC in both and is more stable in the early-warning setting (fold standard deviation 0.039 against 0.074 for the random forest). On still-active customers, the real target of a campaign, it is clearly better in the early-warning version (PR-AUC 0.854 against 0.811, from `tune.py`) and tied in the full version (0.926 against 0.932). In the full version the two models are tied within noise, so the choice there is one of consistency.
"""),
    md(r"""
## E. From scores to a decision

A score is not a decision. The campaign rule contacts a customer when

`churn probability x offer success rate x customer value - offer cost > 0`

with assumptions set in `config/config.yaml` (cost 10, success 30%; units of `customer_value`, which is undocumented). Computed on out-of-fold scores of the tuned gradient boosting (from `python -m churn.evaluate`, seed 42):

| Policy | Customers contacted | Churners reached | Net profit |
|---|---|---|---|
| Contact everyone | 100% | 100% | -12,966 |
| Top 19% by probability (k chosen with hindsight) | 19% | 94% | +11,330 |
| **Expected-gain rule, full model** (applicable) | **12%** | **66%** | **+13,054** |
| Expected-gain rule, early-warning model | 13% | 63% | +12,004 |

The expected-gain rule beats ranking by probability alone because it also uses each customer's value. The hindsight row picks k using the true outcomes, so it is a reference, not a deployable rule.
"""),
    code(r"""
Image("../reports/figures/profit_curve_full.png")
"""),
    md(r"""
## Conclusions

1. A tuned **gradient boosting** reaches a PR-AUC of about 0.95 in the full version (random: 0.16) and finds about 95% of churners in the top 20% of scores.
2. Scores are **not inflated** by duplicate rows.
3. Without the two possibly-late variables, the tuned gradient boosting keeps a PR-AUC of about 0.91 (untuned models: about 0.84), so most of the signal comes from usage.
4. Targeting about **12% of customers** with the expected-gain rule is profitable, while contacting everyone loses money. The profit stays positive across the whole cost and success-rate grid tested (see `evaluate.py`), but in the worst case (10% success, cost 20) it is small (+863, with only 2% of customers contacted), and the assumptions themselves are invented.

Limits: tuned values sit near the edge of the search grid, business assumptions are not measured, and with only 125 still-active churners the active-only metrics are noisy.
"""),
]

if __name__ == "__main__":
    write("02_feature_analysis.ipynb", nb02)
    write("03_model_comparison.ipynb", nb03)