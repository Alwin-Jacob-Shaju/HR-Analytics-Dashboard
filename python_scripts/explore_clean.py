import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import numpy as np

project_root = Path(__file__).parent.parent
raw_file_path = project_root / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
data_dir = project_root / 'data'
reports_dir = project_root / 'reports'


os.makedirs(data_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)


# --- Step 1: Data Loading and Initial Cleaning ---
print(f"Checking for raw CSV at: {raw_file_path}")
if not raw_file_path.exists():
    print(f"Error: File not found at {raw_file_path}")
    exit(1)

try:
    df = pd.read_csv(raw_file_path)
    print("Raw CSV loaded successfully!")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Drop irrelevant/constant columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
print(f"\nColumns after dropping: {len(df.columns)} columns remaining.")

# Save processed data (prior to encoding)
output_path_processed = data_dir / 'processed_data.csv'
df.to_csv(output_path_processed, index=False)
print(f"Processed data (pre-encoding) saved to {output_path_processed}")


# --- Step 2: Analysis and Visualization ---

# Attrition breakdown
print("\nAttrition Breakdown:")
print(df['Attrition'].value_counts(normalize=True))

# Boxplot: Age vs Attrition
plt.figure(figsize=(6, 5))
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title('Age Distribution by Attrition')
plt.savefig(reports_dir / 'age_vs_attrition.png')
plt.close()
print(f"Saved plot to {reports_dir / 'age_vs_attrition.png'}")

# Correlation heatmap (numeric features)
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig(reports_dir / 'correlation_heatmap.png')
plt.close()
print(f"Saved plot to {reports_dir / 'correlation_heatmap.png'}")


# --- Step 3: Feature Engineering (Encoding) ---
le = LabelEncoder()

# Identify all object (string) columns
object_cols = df.select_dtypes(include='object').columns.tolist()

# 3a. Label Encode Binary/Ordinal features (e.g., Target variable and Yes/No flags)
label_encode_cols = ['Attrition', 'Gender', 'OverTime']
nominal_cols = [col for col in object_cols if col not in label_encode_cols]

for col in label_encode_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
        print(f"Label encoded column: {col}")


# 3b. One-Hot Encode Nominal Categorical features
if nominal_cols:
    print(f"\nOne-Hot Encoding {len(nominal_cols)} columns: {nominal_cols}")
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
else:
    print("\nNo remaining nominal categorical columns to One-Hot Encode.")

# Final check of the DataFrame structure
print("\nFinal DataFrame Info after Encoding:")
print(df.info())


# --- Step 4: Save Final Data ---
output_path_final = data_dir / 'final_features.csv'
try:
    df.to_csv(output_path_final, index=False)
    print(f"\nFinal ML-ready data saved to {output_path_final}")
except Exception as e:
    print(f"Error saving final data: {e}")
    exit(1)
