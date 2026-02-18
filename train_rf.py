import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import joblib

# Paths
dataset_path = "predictive_maintenance.csv"
model_dir = "random_forest_model"
model_path = os.path.join(model_dir, "rf_model.joblib")

if not os.path.exists(dataset_path):
    print(f"Error: Dataset {dataset_path} not found.")
    exit(1)

print("Loading data...")
df = pd.read_csv(dataset_path)

# Feature Engineering
print("Feature engineering...")
df['Temperature_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

df = df.rename(columns={
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min'
})

X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
y = df['Target']

type_map = {'L': 0, 'M': 1, 'H': 2}
X['Type'] = X['Type'].map(type_map)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_prob)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {auc_roc:.4f}")

# Save
print(f"Saving model to {model_path}...")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
print("Done.")
