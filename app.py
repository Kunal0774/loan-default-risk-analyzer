# ================================================================
# LOAN DEFAULT RISK ANALYZER — DEPLOYMENT (Flask web app)
# ================================================================
# Loads the model saved by loan_default_analyzer.py and serves a
# form for real-time default-risk prediction.
#
# Run:  python app.py
# Then open http://127.0.0.1:5000
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.json"))

app = Flask(__name__)

FORM_HTML = """
<!doctype html>
<html>
<head>
  <title>Loan Default Risk Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; }
    label { display: block; margin-top: 10px; }
    input, select { width: 100%; padding: 6px; margin-top: 4px; }
    button { margin-top: 20px; padding: 10px 20px; }
    .result { margin-top: 20px; padding: 15px; border-radius: 6px; font-weight: bold; }
    .low { background: #d4edda; color: #155724; }
    .high { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h2>Loan Default Risk Analyzer</h2>
  <form method="POST">
    <label>Loan Amount ($)
      <input type="number" name="loan_amount" value="250000" required>
    </label>
    <label>Applicant Income (monthly, $)
      <input type="number" name="income" value="6000" required>
    </label>
    <label>Property Value ($)
      <input type="number" name="property_value" value="300000" required>
    </label>
    <label>Loan Term (months)
      <input type="number" name="term" value="360" required>
    </label>
    <label>Credit Score
      <input type="number" name="Credit_Score" value="700" min="500" max="900" required>
    </label>
    <label>Loan-to-Value Ratio (LTV, %)
      <input type="number" step="0.1" name="LTV" value="80" required>
    </label>
    <label>Debt-to-Income Ratio (dtir1, %)
      <input type="number" step="0.1" name="dtir1" value="35" required>
    </label>
    <label>Applicant Age
      <select name="age">
        <option value="22"><25</option>
        <option value="30" selected>25-34</option>
        <option value="40">35-44</option>
        <option value="50">45-54</option>
        <option value="60">55-64</option>
        <option value="70">65-74</option>
        <option value="78">>74</option>
      </select>
    </label>
    <label>Gender
      <select name="Gender"><option>Male</option><option>Female</option><option>Joint</option><option>Sex Not Available</option></select>
    </label>
    <label>Credit Worthiness (l1 = higher tier)
      <select name="Credit_Worthiness"><option value="1">l1</option><option value="0">l2</option></select>
    </label>
    <label>Business or Commercial Loan
      <select name="business_or_commercial"><option value="0">No</option><option value="1">Yes</option></select>
    </label>
    <label>Negative Amortization
      <select name="Neg_ammortization"><option value="0">No</option><option value="1">Yes</option></select>
    </label>
    <label>Region
      <select name="Region"><option>south</option><option>North</option><option>central</option><option>North-East</option></select>
    </label>
    <button type="submit">Predict Risk</button>
  </form>
  {% if result %}
    <div class="result {{ 'high' if result.high_risk else 'low' }}">
      Prediction: {{ result.label }}<br>
      Probability of default: {{ result.prob }}%
    </div>
  {% endif %}
</body>
</html>
"""

# Sensible defaults for fields not exposed on the simplified form, based on
# the training data's mode/median — a fuller production form would collect
# every one of these explicitly.
DEFAULTS = {
    "loan_limit": 1, "approv_in_adv": 0, "open_credit": 0,
    "interest_only": 0, "lump_sum_payment": 0, "construction_type": 0,
    "Secured_by": 0, "co-applicant_credit_type": 0,
    "submission_of_application": 0, "Security_Type": 0,
    "total_units_num": 1, "loan_type_type2": 0, "loan_type_type3": 0,
    "loan_purpose_p2": 0, "loan_purpose_p3": 0, "loan_purpose_p4": 0,
    "occupancy_type_pr": 1, "occupancy_type_sr": 0,
    "Region_North-East": 0, "Region_central": 0, "Region_south": 0,
}


def build_features(form):
    loan_amount = float(form["loan_amount"])
    income = float(form["income"])
    property_value = float(form["property_value"])

    row = dict(DEFAULTS)
    row.update({
        "Credit_Worthiness": int(form["Credit_Worthiness"]),
        "business_or_commercial": int(form["business_or_commercial"]),
        "loan_amount": loan_amount,
        "term": float(form["term"]),
        "Neg_ammortization": int(form["Neg_ammortization"]),
        "property_value": property_value,
        "income": income,
        "Credit_Score": float(form["Credit_Score"]),
        "LTV": float(form["LTV"]),
        "dtir1": float(form["dtir1"]),
        "Loan_to_Income": loan_amount / (income + 1),
        "age_numeric": float(form["age"]),
        "loan_amount_Log": np.log1p(loan_amount),
        "income_Log": np.log1p(income),
        "property_value_Log": np.log1p(property_value),
        "Gender_Joint": 1 if form["Gender"] == "Joint" else 0,
        "Gender_Male": 1 if form["Gender"] == "Male" else 0,
        "Gender_Sex Not Available": 1 if form["Gender"] == "Sex Not Available" else 0,
    })

    region = form["Region"]
    row["Region_North-East"] = 1 if region == "North-East" else 0
    row["Region_central"] = 1 if region == "central" else 0
    row["Region_south"] = 1 if region == "south" else 0

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    return df_row


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        X_new = build_features(request.form)

        if type(model).__name__ == "LogisticRegression":
            X_input = scaler.transform(X_new)
        else:
            X_input = X_new

        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        result = {
            "high_risk": bool(pred == 1),
            "label": "HIGH RISK (likely to default)" if pred == 1 else "LOW RISK (likely to repay)",
            "prob": round(prob * 100, 1),
        }

    return render_template_string(FORM_HTML, result=result)


if __name__ == "__main__":
    app.run(debug=True)
