# Loan Default Risk Analyzer

A machine learning project that predicts whether a loan applicant is likely
to **default** (fail to repay) or **repay** their loan, based on their
financial and application details. Includes a working web app for
real-time predictions.

---

## 1. What problem does this solve?

Banks and lenders need to decide: *"If we give this person a loan, how
likely are they to default?"* Instead of a human manually reviewing every
application, this project trains a model on 148,670 real historical loan
records to learn the patterns that separate people who repaid their loans
from people who defaulted — then uses that model to score new applicants
in real time.

---

## 2. The dataset

- **File:** `Loan_Default.csv`
- **Size:** 148,670 loan applications, 34 original columns
- **Target column:** `Status` — `1` = defaulted, `0` = repaid
- **About 1 in 4 loans (24.6%) in this data defaulted.**

Each row is one loan application, with details like the loan amount, the
applicant's income, credit score, property value, region, and so on.

---

## 3. Step-by-step: what the code actually does

### Step 1 — Load the data
Reads the CSV into a table (a pandas DataFrame) and checks its size and
how many loans defaulted vs. didn't.

### Step 2 — Clean the data (and remove "cheating" columns)
Real-world data is messy — some fields are blank, some columns don't
actually help predict the outcome, and some columns *look* helpful but
secretly give away the answer. Here's what was removed and why:

- **`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`** — these
  are only recorded *after* a loan is actually funded. So if a loan
  defaulted, these fields are almost always blank — the model could
  "cheat" by learning "blank field = default" instead of learning real
  risk factors. Removed.
- **`credit_type`** — one specific value in this column predicted default
  with 99.99% accuracy all by itself. That's not a real credit signal;
  it's a quirk in how the dataset was built. A model trained on this would
  look amazing on paper but fail completely on real applicants. Removed.
- **Everything else with missing values** gets filled in — numbers get the
  *median* (the middle value, which resists being skewed by a few extreme
  outliers), and categories get the *most common* value.

### Step 3 — Feature engineering (creating smarter inputs)
Raw columns aren't always the most useful form of the data. A few new,
more informative columns were created:

- **`Loan_to_Income`** — loan amount divided by income. A $300,000 loan
  means something very different to someone earning $200,000/year vs.
  $30,000/year — this ratio captures that directly.
- **`age_numeric`** — age was originally stored as a range like "25-34";
  converted to a single number (30) so the model can use it mathematically.
- **`total_units_num`** — similar cleanup, turning "2U" into the number 2.
- **Log-transformed versions of loan amount, income, and property value** —
  money values are usually "skewed" (most people earn near-average, a few
  earn enormous amounts). A log transform compresses those extreme values
  so they don't unfairly dominate the model.

### Step 4 — Encoding (turning words into numbers)
Machine learning models only understand numbers, not text like "Male" or
"Urban". Two techniques were used:
- **Binary mapping** — for yes/no-style columns (e.g. `Married`: Yes→1, No→0)
- **One-hot encoding** — for columns with several categories (e.g. `Region`
  becomes four separate 0/1 columns, one per region)

### Step 5 — Exploratory Data Analysis (EDA)
Before modeling, the code generates and saves charts to *look at* the data:
- **`target_distribution.png`** — how many loans defaulted vs. didn't
- **`correlation_heatmap.png`** — which numeric features move together

### Step 6 — Train/test split
The data is split: **80% to train** the model (teach it patterns) and
**20% to test** it (check how well it performs on data it's never seen —
this is the only fair way to measure real-world performance).

### Step 7 — Train three different models
Three different algorithms were trained and compared, so the choice of
"best model" is backed by evidence, not guesswork:

| Model | What it is, in plain terms |
|---|---|
| **Logistic Regression** | The simplest approach — draws a straight-line-style boundary between "likely to default" and "likely to repay." Fast and easy to explain, but can't capture complex patterns. |
| **Random Forest** | Builds hundreds of decision trees (like a flowchart of yes/no questions), each trained slightly differently, then averages their votes. Better at capturing complex, non-linear patterns. |
| **XGBoost** | Also builds many decision trees, but each new tree is trained specifically to fix the mistakes of the trees before it. Usually the strongest performer on this kind of structured, spreadsheet-style data. |

### Step 8 — Evaluate each model
Three metrics were used to judge each model, because no single number
tells the whole story:

- **Accuracy** — % of predictions that were correct overall. Can be
  misleading when one outcome (repaid) is much more common than the other
  (defaulted) — a model could get 75% "accuracy" just by always guessing
  "repaid" and never actually learning anything.
- **F1-Score** — balances how many actual defaulters the model catches
  (recall) against how often it cries wolf on someone who was actually
  fine (precision). More trustworthy than accuracy on imbalanced data
  like this.
- **ROC-AUC** — how well the model *ranks* risky applicants above safe
  ones, on a scale from 0.5 (random guessing) to 1.0 (perfect). This is
  the primary metric for a risk-scoring problem like this one, because a
  lender usually wants a *risk score* to set their own approval cutoff,
  not just a flat yes/no.

### Step 9 — Pick the winner and explain it
The model with the best ROC-AUC is automatically selected and saved. A
chart of the ROC curves (`roc_curve_comparison.png`) and a bar chart of
which features mattered most (`feature_importance.png`) are also saved,
so you can see *why* the model makes the decisions it does — not just
that it works.

### Step 10 — Save the model for reuse
The winning model, the scaler (used to normalize numbers), and the exact
list of feature columns are all saved to the `saved_model/` folder using
`joblib`. This means the model doesn't need to be retrained every time —
it can be loaded instantly and reused, which is what the web app does.

---

## 4. Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **XGBoost (winner)** | 90.1% | 0.763 | **0.899** |
| Random Forest | 88.7% | 0.711 | 0.885 |
| Logistic Regression | 80.3% | 0.442 | 0.764 |

**XGBoost was selected** as the final model — it had the best ROC-AUC and
by far the best F1-Score, meaning it's the most reliable at actually
telling risky and safe applicants apart, not just guessing the majority
class.

The strongest predictors of default turned out to be: `lump_sum_payment`,
`property_value`, `LTV` (loan-to-value ratio), `Neg_ammortization`, and
`Credit_Worthiness` — all genuine underwriting factors, spread across
several features rather than one column dominating everything (which
would have been a red flag for leftover leakage).

---

## 5. The web app (`app.py`)

A simple Flask web application that loads the saved model and lets anyone
enter an applicant's details into a form — loan amount, income, credit
score, property value, etc. — and instantly see:
- A **risk label**: "LOW RISK (likely to repay)" or "HIGH RISK (likely to
  default)"
- A **probability**, e.g. "12.4% chance of default"

This is the "deployment" piece — turning a trained model sitting in a
file into something an actual user could interact with.

### How to run it
```bash
pip install -r requirements.txt
python loan_default_analyzer.py   # trains the model and creates saved_model/
python app.py                     # starts the web app
```
Then open **http://127.0.0.1:5000** in your browser.

---

## 6. Project files

```
loan_default_analyzer.py   → full training pipeline (cleaning → features → 3 models → evaluation → save)
app.py                     → Flask web app for real-time predictions
requirements.txt           → Python packages needed
saved_model/
  ├── best_model.pkl              → the trained XGBoost model
  ├── scaler.pkl                  → normalizes numeric inputs (used if the best model is Logistic Regression)
  ├── feature_columns.json        → exact list/order of features the model expects
  ├── model_comparison.csv        → the metrics table above, as a CSV
  ├── roc_curve_comparison.png    → ROC curves for all 3 models
  ├── feature_importance.png      → which features drove the predictions
  ├── correlation_heatmap.png     → EDA chart
  └── target_distribution.png     → EDA chart
```

---

## 7. Honest limitations (worth knowing, not hiding)

- This dataset labels **default vs. repaid**, based on historical US
  mortgage-style loan data — results won't automatically transfer to a
  different lending market without retraining on relevant data.
- Two columns were deliberately removed because they leaked the answer
  (explained in Step 2) — a reminder that a model that looks *too* good
  is worth double-checking before trusting it.
- The simplified web form doesn't expose every one of the 39 features the
  model actually uses — some are set to sensible defaults for
  demonstration purposes. A production version would collect all of them.
