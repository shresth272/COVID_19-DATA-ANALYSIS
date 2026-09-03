# Week 4 - COVID-19 Machine Learning Model
# Logistic Regression classification with model evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)

# 1. Load the cleaned dataset from your GitHub project
df = pd.read_csv("Covid Data cleaned.csv")

# 2. Convert the target into binary classification
# Your existing project creates CLASIFFICATION_FINAL with Positive/Negative.
target = "CLASIFFICATION_FINAL"
df = df[df[target].notna()].copy()

# Keep only the two expected classes
df[target] = df[target].astype(str).str.strip()
df = df[df[target].isin(["Positive", "Negative"])].copy()
y = (df[target] == "Positive").astype(int)

# 3. Select useful predictors already present in your project.
candidate_features = [
    "AGE", "SEX", "PATIENT_TYPE", "PNEUMONIA", "DIABETES", "COPD",
    "ASTHMA", "HIPERTENSION", "CARDIOVASCULAR", "OBESITY",
    "RENAL_CHRONIC", "TOBACCO", "USMER", "MEDICAL_UNIT"
]
features = [c for c in candidate_features if c in df.columns]

X = df[features].copy()

# Convert numeric columns to numeric where possible
for c in ["AGE", "MEDICAL_UNIT"]:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Make categorical columns strings so OneHotEncoder can process them
categorical_cols = [c for c in X.columns if c not in ["AGE", "MEDICAL_UNIT"]]
numeric_cols = [c for c in X.columns if c in ["AGE", "MEDICAL_UNIT"]]

# 4. Use a reproducible sample when the dataset is extremely large.
# This keeps the internship model fast while preserving a representative sample.
MAX_ROWS = 100000
if len(X) > MAX_ROWS:
    sample_idx = df.sample(MAX_ROWS, random_state=42).index
    X = X.loc[sample_idx]
    y = y.loc[sample_idx]

print("Rows used:", len(X))
print("Features:", features)
print("Class distribution:")
print(y.value_counts())

# 5. Preprocessing
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 7. Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=300, class_weight="balanced"))
])

model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n--- MODEL RESULTS ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0
))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title("COVID-19 Classification - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("week4_confusion_matrix.png", dpi=200)
plt.show()

# 11. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - COVID-19 Classification")
plt.legend()
plt.tight_layout()
plt.savefig("week4_roc_curve.png", dpi=200)
plt.show()

# 12. Save a compact results file for the report
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, auc]
})
results.to_csv("week4_model_results.csv", index=False)

print("\nFiles created:")
print("- week4_confusion_matrix.png")
print("- week4_roc_curve.png")
print("- week4_model_results.csv")
