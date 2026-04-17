# ================================
# LOAN DEFAULT RISK ANALYZER (FINAL CLEAN CODE)
# ================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# 2. LOAD DATASET (IMPORTANT)
# ================================
path = r"E:\KUNAL!!!\PROJECTS!!!!\loan risk analyzer\train.csv"

print("Checking file path...")
print("File exists:", os.path.exists(path))   # MUST be True

if not os.path.exists(path):
    raise Exception(" File not found. Check your path!")

df = pd.read_csv(path)

print("\nFirst 5 rows:\n", df.head())
print("\nColumns:\n", df.columns)

# ================================
# 3. DATA CLEANING
# ================================

# Drop unnecessary column
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Remove duplicates
df = df.drop_duplicates()

# ================================
# 4. ENCODE CATEGORICAL DATA
# ================================
le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ================================
# 5. EDA (VISUALIZATION)
# ================================
plt.figure()
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Default Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 6. FEATURE & TARGET
# ================================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Convert Y/N → 0/1
y = y.map({'Y': 0, 'N': 1}) if y.dtype == 'object' else y

# ================================
# 7. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 8. MODEL BUILDING
# ================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# ================================
# 9. PREDICTIONS
# ================================
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ================================
# 10. EVALUATION
# ================================
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Report:\n", classification_report(y_test, lr_pred))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Report:\n", classification_report(y_test, rf_pred))

# ================================
# 11. TEST NEW CUSTOMER
# ================================
sample = X_test.iloc[0].values.reshape(1, -1)

prediction = rf_model.predict(sample)

print(" NEW CUSTOMER RESULT ")
if prediction[0] == 1:
    print(" HIGH RISK (May Default)")
else:
    print("SAFE CUSTOMER")