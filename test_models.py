import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# 1. Paths
dataset_path = "predictive_maintenance.csv"
xgb_path = "xgboost_model/xgb_model.joblib"
rf_path = "random_forest_model/rf_model.joblib"
lr_path = "logistic_regression_model/lr_model.joblib"
scaler_path = "logistic_regression_model/scaler.joblib"

# 2. Load Models
print("Loading models...")
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)
lr_model = joblib.load(lr_path)
lr_scaler = joblib.load(scaler_path)

# 3. Load and Preprocess Data
print("Loading and preprocessing data...")
df = pd.read_csv(dataset_path)

# Feature Engineering
df['Temperature_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# Clean feature names (must match training)
df = df.rename(columns={
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min'
})

# Select features
X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
y = df['Target']

# Encode 'Type'
type_map = {'L': 0, 'M': 1, 'H': 2}
X['Type'] = X['Type'].map(type_map)

# Split (using same seed as training for comparison)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Predictions
print("\nRunning predictions...")

# XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Random Forest
y_pred_rf = rf_model.predict(X_test)

# Logistic Regression (needs scaling)
X_test_scaled = lr_scaler.transform(X_test)
y_pred_lr = lr_model.predict(X_test_scaled)

# 5. Summary Report
test_results = {}

def get_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\n--- {name} Report ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

test_results["xgboost"] = get_metrics("XGBoost", y_test, y_pred_xgb)
test_results["random_forest"] = get_metrics("Random Forest", y_test, y_pred_rf)
test_results["logistic_regression"] = get_metrics("Logistic Regression", y_test, y_pred_lr)

# 6. Joint Prediction Analysis
results_df = pd.DataFrame({
    'Actual': y_test,
    'XGBoost': y_pred_xgb,
    'RandomForest': y_pred_rf,
    'LogisticReg': y_pred_lr
})

print("\n--- Model Agreement ---")
agreement_data = {}
models = ['XGBoost', 'RandomForest', 'LogisticReg']
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        agreement = (results_df[models[i]] == results_df[models[j]]).mean()
        key = f"{models[i]}_vs_{models[j]}"
        agreement_data[key] = float(agreement)
        print(f"Agreement between {models[i]} and {models[j]}: {agreement:.2%}")

test_results["model_agreement"] = agreement_data

# 7. Save to JSON
output_json = "final_test_results.json"
with open(output_json, 'w') as f:
    json.dump(test_results, f, indent=4)

print(f"\nResults saved to {output_json}")
print("\nTesting complete.")
