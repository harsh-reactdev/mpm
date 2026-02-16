import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/harshithmr/Desktop/Workspace/mpm/predictive_maintenance.csv")

print("Columns:", df.columns.tolist())
print(f"Total rows: {len(df)}")

print("\nData Types:")
print(df.dtypes)

print("\nValue counts for Target:")
print(df['Target'].value_counts())

print("\nValue counts for Failure Type:")
print(df['Failure Type'].value_counts())

print("\nNunique for Type:")
print(df['Type'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
