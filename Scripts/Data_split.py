import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent
clean_data_dir = script_dir / "Data" / "clean"
split_data_dir = script_dir / "Data" / "split_data"
# 1. Load the cleaned data
df = pd.read_csv(clean_data_dir / "Clean_data_for_train(1).csv")
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

print("Train-Test split done.")
print(f"Train samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")
print(f"Train samples: {Y_train.shape}")
print(f"Test samples: {Y_test.shape}")

X_train.to_csv(split_data_dir / "X_train.csv", index=False)
X_test.to_csv(split_data_dir / "X_test.csv", index=False)
Y_train.to_csv(split_data_dir / "Y_train.csv", index=False)
Y_test.to_csv(split_data_dir / "Y_test.csv", index=False)

# Save features
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, split_data_dir / 'feature_columns.pkl')