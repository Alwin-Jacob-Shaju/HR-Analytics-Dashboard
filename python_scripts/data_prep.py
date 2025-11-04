import pandas as pd
import numpy as np
import datetime as dt

# --- Configuration ---
FILE_PATH = 'C:/Users/M S I/Desktop/my projects/HR Employee Attrition Prediction/WA_Fn-UseC_-HR-Employee-Attrition.csv'
OUTPUT_FILE = 'HR_Cleaned_for_PBI.csv'

# 1. Load Data
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Original shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please check the path.")
    exit()

# 2. Data Cleaning and Redundancy Removal
# Drop columns that are constant or unnecessary for analysis
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df.drop(columns=columns_to_drop, inplace=True)
print(f"Shape after dropping redundant columns: {df.shape}")

# 3. Feature Engineering: Creating Dimensions for Analysis

# a. Attrition Flag (for easy calculation)
df['Is_Attrited'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# b. Age Group
bins = [18, 25, 35, 45, 55, 60]
labels = ['18-24', '25-34', '35-44', '45-54', '55+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# c. Tenure Group
# We use YearsAtCompany as a proxy for tenure since we don't have HireDate/SeparationDate
tenure_bins = [0, 2, 5, 10, 20, 100]
tenure_labels = ['<2 Years', '2-5 Years', '6-10 Years', '11-20 Years', '20+ Years']
df['Tenure_Group'] = pd.cut(df['YearsAtCompany'], bins=tenure_bins, labels=tenure_labels, right=False)

# d. Satisfaction Index (combining multiple satisfaction scores)
# Create a simple average index
satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                     'RelationshipSatisfaction', 'WorkLifeBalance']
df['Overall_Satisfaction_Score'] = df[satisfaction_cols].mean(axis=1).round(1)

# e. Performance Category (based on Rating)
df['Performance_Category'] = df['PerformanceRating'].apply(
    lambda x: 'High' if x == 4 else ('Medium' if x == 3 else 'Low')
)

# 4. Export the Cleaned Dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Data preparation complete. Cleaned file saved as: {OUTPUT_FILE}")

# 5. Quick Attrition Rate Check (Sanity Check)
total_employees = len(df)
attrition_count = df['Is_Attrited'].sum()
attrition_rate = (attrition_count / total_employees) * 100

print(f"\n--- Sanity Check ---")
print(f"Total Employees: {total_employees}")
print(f"Overall Attrition Rate: {attrition_rate:.2f}%")
print("--------------------")