# ================================================================
# LOAN DEFAULT RISK ANALYZER  —  END-TO-END ML PIPELINE
# ================================================================
# Dataset  : Loan_Default.csv (148,670 real loan records, US mortgage-style data)
# Target   : Status  (1 = Defaulted, 0 = Did not default)
# Models   : Logistic Regression, Random Forest, XGBoost
# Metrics  : Accuracy, F1-Score, ROC-AUC, Confusion Matrix
# Output   : Saved model + scaler + encoders for deployment (app.py)
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Loan_Default.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

# ----------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Loan_Default.csv not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Default rate: {df['Status'].mean():.1%}")

# ----------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# ID and year add no predictive value (year is a single constant value)
df = df.drop(columns=["ID", "year"])

# --- IMPORTANT: LEAKAGE CHECK ---
# rate_of_interest, Interest_rate_spread and Upfront_charges are missing for
# ~99.5% of defaulted loans but present for ~99.7% of non-defaulted loans.
# These fields only get populated once a loan is actually originated/funded,
# so their presence effectively reveals the outcome we're trying to predict.
# A model trained on these would look artificially perfect and be useless
# in production (that information doesn't exist yet at application time).
# They are dropped here for a realistic, deployable model.
leaky_cols = ["rate_of_interest", "Interest_rate_spread", "Upfront_charges"]
print(f"Dropping leaky post-origination columns: {leaky_cols}")
df = df.drop(columns=leaky_cols)

# --- SECOND LEAKAGE CHECK ---
# credit_type == 'EQUI' predicts Status == 1 (default) with ~99.99% certainty
# (15,298 rows, virtually all defaults). That's not a real credit signal —
# it's a structural artifact of how this dataset was assembled (this bureau
# code appears to only get used for a specific loan-outcome subset). Keeping
# it would make the model look artificially excellent while learning nothing
# generalizable, and it would collapse the moment a real applicant's credit
# report doesn't fit that pattern. Dropped for a trustworthy model.
print("Dropping 'credit_type' — near-perfect but spurious correlation with target")
df = df.drop(columns=["credit_type"])

print("\nMissing values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# property_value, LTV and dtir1 are also missing far more often for defaults
# (~41-45%) than non-defaults (~0-7%) — a softer version of the same leakage.
# An explicit "was this missing" flag was tested here and turned out to
# single-handedly dominate the model (~75% of feature importance) — i.e. the
# model was just re-learning the leak through the back door. So instead we
# simply median-impute these fields and let the model use only the (mostly
# reasonable) imputed values, not the fact that they were missing.
# Fill missing values: median for numeric, mode for categorical
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()
print(f"\nShape after cleaning: {df.shape}")

# ----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# Loan amount relative to income — core underwriting affordability ratio
df["Loan_to_Income"] = df["loan_amount"] / (df["income"] + 1)

# Age bracket -> numeric midpoint, so the model can use it as an ordered value
age_map = {"<25": 22, "25-34": 30, "35-44": 40, "45-54": 50,
           "55-64": 60, "65-74": 70, ">74": 78}
df["age_numeric"] = df["age"].map(age_map)

# total_units "1U".."4U" -> integer
df["total_units_num"] = df["total_units"].str.replace("U", "", regex=False).astype(int)

# Log-transform skewed monetary columns
for col in ["loan_amount", "income", "property_value"]:
    df[col + "_Log"] = np.log1p(df[col].clip(lower=0))

print("New features: Loan_to_Income, age_numeric, total_units_num, "
      "and log-transforms of loan_amount/income/property_value")

# ----------------------------------------------------------------
# 4. ENCODE CATEGORICAL VARIABLES
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: ENCODING")
print("=" * 60)

binary_maps = {
    "loan_limit": {"cf": 1, "ncf": 0},
    "approv_in_adv": {"pre": 1, "nopre": 0},
    "Credit_Worthiness": {"l1": 1, "l2": 0},
    "open_credit": {"opc": 1, "nopc": 0},
    "business_or_commercial": {"b/c": 1, "nob/c": 0},
    "Neg_ammortization": {"neg_amm": 1, "not_neg": 0},
    "interest_only": {"int_only": 1, "not_int": 0},
    "lump_sum_payment": {"lpsm": 1, "not_lpsm": 0},
    "construction_type": {"mh": 1, "sb": 0},
    "Secured_by": {"land": 1, "home": 0},
    "co-applicant_credit_type": {"EXP": 1, "CIB": 0},
    "submission_of_application": {"to_inst": 1, "not_inst": 0},
    "Security_Type": {"Indriect": 1, "direct": 0},
}
for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

# Multi-category columns -> one-hot
onehot_cols = ["Gender", "loan_type", "loan_purpose", "occupancy_type", "Region"]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

# Drop original string columns already converted to numeric equivalents
df = df.drop(columns=["age", "total_units"])

print("Encoding complete. Total features:", df.shape[1] - 1)

# ----------------------------------------------------------------
# 5. EDA (saved as image files)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

plt.figure(figsize=(5, 4))
sns.countplot(x="Status", data=df)
plt.title("Target Distribution (0=No Default, 1=Default)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "target_distribution.png"))
plt.close()

# Correlation heatmap on a manageable subset (too many one-hot columns to plot all)
key_cols = ["Status", "loan_amount", "income", "Credit_Score", "LTV", "dtir1",
            "Loan_to_Income", "age_numeric", "term"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[key_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (key numeric features)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "correlation_heatmap.png"))
plt.close()

print("Saved EDA charts to:", MODEL_DIR)

# ----------------------------------------------------------------
# 6. FEATURES & TARGET
# ----------------------------------------------------------------
X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------
# 7. MODEL TRAINING — 3 ALGORITHMS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: TRAINING 3 MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE
    ),
}

results = []
fitted_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "ROC-AUC": auc})
    fitted_models[name] = model

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("Classification Report:\n", classification_report(y_test, pred))

# ----------------------------------------------------------------
# 8. MODEL COMPARISON
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model by ROC-AUC: {best_model_name}")

# ROC curve comparison plot
plt.figure(figsize=(6, 5))
for name, model in fitted_models.items():
    if name == "Logistic Regression":
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "roc_curve_comparison.png"))
plt.close()

# Feature importance for the best tree-based model (or RF as fallback)
importance_source = best_model if best_model_name != "Logistic Regression" else fitted_models["Random Forest"]
importances = pd.Series(importance_source.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
importances.head(12).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 12 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))
plt.close()

print("\nTop 8 most important features:\n", importances.head(8))

# ----------------------------------------------------------------
# 9. SAVE MODEL FOR DEPLOYMENT
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: SAVING MODEL ARTIFACTS")
print("=" * 60)

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.json"))
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

print(f"Saved: best_model.pkl ({best_model_name}), scaler.pkl, feature_columns.json")

# ----------------------------------------------------------------
# 10. TEST ON A NEW / SAMPLE CUSTOMER
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9: SAMPLE PREDICTION")
print("=" * 60)

sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)

if best_model_name == "Logistic Regression":
    pred = best_model.predict(sample_scaled)[0]
    prob = best_model.predict_proba(sample_scaled)[0][1]
else:
    pred = best_model.predict(sample)[0]
    prob = best_model.predict_proba(sample)[0][1]

print(f"Predicted class: {'HIGH RISK (likely to default)' if pred == 1 else 'LOW RISK (likely to repay)'}")
print(f"Predicted probability of default: {prob:.2%}")

print("\nPipeline complete.")
