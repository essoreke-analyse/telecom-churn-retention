"""Apply the text corrections to tools/make_notebooks.py (run once)."""
from pathlib import Path

path = Path(__file__).resolve().parent / "make_notebooks.py"
text = path.read_text(encoding="utf-8")

REPLACEMENTS = [
    # 1. usage bullet + call_failure bullet
    (
        """- **Usage dominates.** `frequency_of_use` is the top variable in both versions. Customers who call less are the ones who leave.
- **`call_failure` and `subscription_length` are the surprise.** Alone they barely separate churners (univariate power close to 0.5), yet they weigh heavily once usage is controlled. Heavy users have more failed calls but leave less, so the effect only shows up *conditionally*. Dropping them""",
        """- **Usage dominates.** `frequency_of_use` is the top variable in both versions, and `seconds_of_use` and `customer_value` are the best single separators. Customers who use the service less are the ones who leave. Because these usage variables are strongly correlated, the ranking among them should not be over-interpreted.
- **`call_failure` and `subscription_length` are the surprise.** Alone they barely separate churners (univariate power close to 0.5), yet they weigh heavily once usage is controlled. `call_failure` is correlated with usage (about 0.5), so its effect is probably only visible *conditionally*; this is a plausible reading, not something these data prove. Dropping them""",
    ),
    # 2. charge_band bullet, added before the caution bullet
    (
        """- **Caution.** Usage variables are strongly correlated""",
        """- **`charge_band` is useless in the full model but useful in the early-warning one** (0.048), where it partly substitutes for the removed variables.
- **Caution.** Usage variables are strongly correlated""",
    ),
    # 3. note on empty cells
    (
        """which may be consequences of leaving (the dataset does not say when they were measured).
\"\"\"),
    code(r\"\"\"
uni = {}""",
        """which may be consequences of leaving (the dataset does not say when they were measured).

*Univariate power is computed for numeric variables only (Mann-Whitney). For binary and categorical variables, the association with churn is shown in notebook 01 (Cramér's V). That is why some cells are empty.*
\"\"\"),
    code(r\"\"\"
uni = {}""",
    ),
    # 4. complaints x inactivity comment
    (
        '''Active customers who did not complain are the hardest to spot yet still contain a meaningful number of churners. A rule such as "contact those who complain" would miss most departures, which is why the model must rely on usage.''',
        '''Active customers who did not complain are the hardest to spot: they still contain 66 churners among 2,285 customers (2.9%), more than the 59 churners among active complainers. A rule such as "contact those who complain" would miss most departures, which is why the model must rely on usage.''',
    ),
    # 5. conclusion 1
    (
        "1. Usage is the strongest and most actionable signal, and it is observable without any complaint.",
        "1. Usage is the strongest behavioural signal, and it is observable without any complaint.",
    ),
]

for old, new in REPLACEMENTS:
    assert old in text, f"Text not found, nothing written:\n{old[:80]}..."
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print(f"{len(REPLACEMENTS)} corrections applied to {path.name}")