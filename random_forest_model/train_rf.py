import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib

# 1. Setup
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Load and Preprocess
df = pd.read_csv("../predictive_maintenance.csv")

# Feature Engineering
df['Temperature_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# Clean feature names
df = df.rename(columns={
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min'
})

# Drop columns
X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
y = df['Target']

# Encode 'Type'
type_map = {'L': 0, 'M': 1, 'H': 2}
X['Type'] = X['Type'].map(type_map)

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Model
print("Training Random Forest with class_weight='balanced'...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metric Calculations
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)

# Specificity Calculation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
spec = tn / (tn + fp)

metrics = {
    "Accuracy": float(acc),
    "Precision": float(prec),
    "Recall": float(rec),
    "F1_Score": float(f1),
    "AUC_ROC": float(auc_roc),
    "Specificity": float(spec)
}

print("\nMetrics (Random Forest):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save metrics to JSON
with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# 6. Visualizations

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Precision-Recall Curve
p, r, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(r, p, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
plt.close()

# 7. Save Model
print("Saving Random Forest model...")
joblib.dump(model, 'rf_model.joblib')

print(f"\nAll results saved in {output_dir}/")
