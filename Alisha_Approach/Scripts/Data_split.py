import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# 1. Load the cleaned data
df = pd.read_csv("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/clean/Clean_data_for_train(1).csv")
print(df.columns)

# 2. Define features (X) and target (y)
target = 'readmitted_30days'
features = df.drop(columns=[target])
X = features
y = df[target]

# 3. Train-Test Split (stratified)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# print("Train-Test split done.")
# print(f"Train samples: {X_train.shape}")
# print(f"Test samples: {X_test.shape}")
# print(f"Train samples: {Y_train.shape}")
# print(f"Test samples: {Y_test.shape}")

# 4. Optional: Save to files if needed
# X_train.to_csv("Data/split_data/X_train.csv", index=False)
# X_test.to_csv("Data/split_data/X_test.csv", index=False)
# Y_train.to_csv("Data/split_data/Y_train.csv", index=False)
# Y_test.to_csv("Data/split_data/Y_test.csv", index=False)

# Save features
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, '/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/split_data/feature_columns.pkl')