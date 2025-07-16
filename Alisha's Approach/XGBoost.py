import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt 
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (precision_recall_curve, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score )

# Read data
x_train = pd.read_csv('Data/X_train.csv')
x_test = pd.read_csv('Data/X_test.csv')
y_train = pd.read_csv('Data/Y_train.csv')
y_test = pd.read_csv('Data/Y_test.csv')

# Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# SMOTE
smote = SMOTE(random_state = 42)
x_train_res, y_train_res = smote.fit_resample(x_train_scaled, y_train.values.ravel())

# XGBoost with best parameters
xgb = XGBClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    scale_pos_weight=2,  # Adjust for imbalance: pos_weight = (neg/pos)
    eval_metric='logloss'
)
xgb.fit(x_train_res, y_train_res)

# Predict probabilities
y_probs_xgb = xgb.predict_proba(x_test_scaled)[:, 1]

# Find best threshold (optional, here we use 0.25)
final_threshold = 0.25
y_pred_xgb = (y_probs_xgb > final_threshold).astype(int)

# Metrics
print("=== Evaluation at Threshold 0.25 ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_probs_xgb))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'XGBoost Confusion Matrix at Threshold: {final_threshold}')
plt.show()

#########################################
# Precision, Recall, F1 vs. Threshold Curve
thresholds_to_check = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
precisions, recalls, f1_scores = [], [], []

for thresh in thresholds_to_check:
    preds = (y_probs_xgb > thresh).astype(int)
    precisions.append(precision_score(y_test, preds, zero_division=0))
    recalls.append(recall_score(y_test, preds, zero_division=0))
    f1_scores.append(f1_score(y_test, preds, zero_division=0))
    print(f"Threshold {thresh:.2f} - Precision: {precisions[-1]:.3f}, Recall: {recalls[-1]:.3f}, F1: {f1_scores[-1]:.3f}")

# Plot precision, recall, F1 vs threshold
plt.figure(figsize=(10,6))
plt.plot(thresholds_to_check, precisions, marker='o', label='Precision')
plt.plot(thresholds_to_check, recalls, marker='o', label='Recall')
plt.plot(thresholds_to_check, f1_scores, marker='o', label='F1 Score')
plt.axvline(final_threshold, color='red', linestyle='--', label=f'Chosen Threshold = {final_threshold}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, F1 vs Threshold')
plt.legend()
plt.grid()
plt.show()


###### Feature Importance 
importances = xgb.feature_importances_
features = x_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_n = 15
top_features = importance_df.nlargest(15, 'Importance').copy()

# Apply to Feature column
top_features['Feature'] = top_features['Feature'].str.replace('discharge_disposition_', 'dd_', regex=False)

def shorten_name(name, max_len=25):
    return name if len(name) <= max_len else name[:max_len] + '...'

# Apply to Feature column
top_features['Feature'] = top_features['Feature'].apply(shorten_name)

plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title("XGBoost Top 15 Feature Importance", fontsize=16, pad=20)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
print(top_features)
