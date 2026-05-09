# ============================================
# DATA PREPROCESSING - WALMART DATASET
# ============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("preprocess_data_walmart_dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Separate categorical columns
categorical_columns = df.select_dtypes(include='object').columns

# Label Encoding
label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nEncoded Dataset:")
print(df.head())

# Feature Scaling
scaler = StandardScaler()

# Selecting feature columns
X = df.iloc[:, :-1]

# Target column
y = df.iloc[:, -1]

# Scale features
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Convert scaled data back to dataframe
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Save preprocessed data
X_scaled_df.to_csv("preprocessed_data.csv", index=False)

print("\nData preprocessing completed successfully.")
