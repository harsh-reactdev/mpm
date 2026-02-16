import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Set plotting style
sns.set_theme(style="whitegrid")

# Create output directory for figures if it doesn't exist
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Load the dataset
print("Loading dataset...")
file_path = "/Users/harshithmr/Desktop/Workspace/mpm/predictive_maintenance.csv"
df = pd.read_csv(file_path)

# 2. Preprocessing
print("Preprocessing data...")
# Drop identifiers
df_base = df.drop(['UDI', 'Product ID'], axis=1)

# Encode 'Type' (L, M, H)
type_map = {'L': 0, 'M': 1, 'H': 2}
df_base['Type'] = df_base['Type'].map(type_map)

# 3. Feature Engineering
print("Engineering new features...")
df_base['Temperature_Difference'] = df_base['Process temperature [K]'] - df_base['Air temperature [K]']
df_base['Power'] = df_base['Rotational speed [rpm]'] * df_base['Torque [Nm]']

# Function to calculate and plot importance
def analyze_importance(df_input, target_col, leakage_col, title, filename):
    print(f"Analyzing importance for: {target_col}...")
    
    # Drop target and any leakage column
    X = df_input.drop([target_col, leakage_col], axis=1)
    y = df_input[target_col]
    
    # Handle categorical target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Encoded classes for {target_col}: {le.classes_}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    return importance_df

# 4. Binary Analysis (Target)
binary_importance = analyze_importance(
    df_base, 'Target', 'Failure Type', 
    "Feature Importance: Binary Failure (Target)", 
    "binary_feature_importance.png"
)

# 5. Multi-class Analysis (Failure Type)
multiclass_importance = analyze_importance(
    df_base, 'Failure Type', 'Target', 
    "Feature Importance: Specific Failure Types", 
    "multiclass_feature_importance.png"
)

# 6. Correlation Analysis (on features only)
print("Calculating correlations...")
features_only = df_base.drop(['Target', 'Failure Type'], axis=1)
plt.figure(figsize=(12, 10))
corr = features_only.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

print("\nBinary Importance:")
print(binary_importance)
print("\nMulti-class Importance:")
print(multiclass_importance)

print(f"\nAnalysis complete. Results saved in '{output_dir}/'")
