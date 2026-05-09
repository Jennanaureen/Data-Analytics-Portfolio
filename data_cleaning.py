# ============================================
# DATA CLEANING - WALMART DATASET
# ============================================

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill numerical missing values with mean
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Remove outliers using IQR method
for column in numeric_columns:
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower_limit) &
        (df[column] <= upper_limit)
    ]

# Check cleaned data
print("\nCleaned Dataset Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("final_cleaned_data.csv", index=False)

print("\nData cleaning completed successfully.")
