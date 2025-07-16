import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer

# Read data
x_train = pd.read_csv('Data/X_train.csv')
x_test = pd.read_csv('Data/X_test.csv')
y_train = pd.read_csv('Data/Y_train.csv')
y_test = pd.read_csv('Data/Y_test.csv')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# SMOTE
smote = SMOTE(random_state = 42)
x_train_res, y_train_res = smote.fit_resample(x_train_scaled, y_train.values.ravel())

#### Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(x_train_res, y_train_res)

y_probs = rf.predict_proba(x_test_scaled)[:, 1]


#########################################
# Evaluate at Fixed Threshold = 0.20
final_threshold = 0.20
y_pred_thresh = (y_probs > final_threshold).astype(int)

# Metrics
print("=== Evaluation at Threshold 0.20 ===")
print("AUC:", roc_auc_score(y_test, y_probs))
print("F1-score:", f1_score(y_test, y_pred_thresh))
print("Precision:", precision_score(y_test, y_pred_thresh))
print("Recall:", recall_score(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))

cm = confusion_matrix(y_test, y_pred_thresh)
sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
plt.title(f'Random Forest Confusion Matrix at Threshold: {final_threshold}')
plt.show()

#########################################
# Precision, Recall, F1 vs. Threshold Curve
thresholds_to_check = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
precisions, recalls, f1_scores = [], [], []

for thresh in thresholds_to_check:
    preds = (y_probs > thresh).astype(int)
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


#### Feature Importance
importances = rf.feature_importances_
features = x_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_n = 15
top_features = importance_df.head(top_n)

plt.figure(figsize=(10,8))
#print(importance_df)
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title(f"Top {top_n} Important Feature for Random Forest")
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
print(top_features)
