import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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

log_reg = LogisticRegression(max_iter=2000,  class_weight= 'balanced')
log_reg.fit(x_train_res, y_train_res)

y_prob = log_reg.predict_proba(x_test_scaled)[:,1]

#########################################
# Evaluate at Fixed Threshold = 0.6
threshold = 0.6
y_pred_thresh = (y_prob > threshold).astype(int)

# Metrics
print("=== Evaluation at Threshold 0.6 ===")
print("AUC:", roc_auc_score(y_test, y_prob))
print("F1-score:", f1_score(y_test, y_pred_thresh))
print("Precision:", precision_score(y_test, y_pred_thresh))
print("Recall:", recall_score(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_thresh)
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Logistic Regression Confusion Matrix at Threshold: {threshold}')
plt.tight_layout()
plt.show()

# #########################################
# Precision, Recall, F1 vs. Threshold Curve
thresholds_to_check = np.arange(0.1, 0.91, 0.1)
precisions = []
recalls = []
f1_scores = []

for thresh in thresholds_to_check:
    y_pred = (y_prob > thresh).astype(int)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

plt.figure(figsize=(8,6))
plt.plot(thresholds_to_check, precisions, label='Precision')
plt.plot(thresholds_to_check, recalls, label='Recall')
plt.plot(thresholds_to_check, f1_scores, label='F1 Score')
plt.axvline(threshold, color='red', linestyle='--', label=f'Chosen Threshold = {threshold}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, F1 vs Threshold')
plt.legend()
plt.grid()
plt.show()

#### Feature Importance
importance = np.abs(log_reg.coef_[0])
features = x_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_n = 15
top_features = importance_df.nlargest(15, 'Importance').copy()

# Apply to Feature column
top_features['Feature'] = top_features['Feature'].str.replace('discharge_disposition_', 'dd_', regex=False)

def shorten_name(name, max_len=50):
    return name if len(name) <= max_len else name[:max_len] + '...'

# Apply to Feature column
top_features['Feature'] = top_features['Feature'].apply(shorten_name)

plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title(f"Top {top_n} Important Feature for Logistic Regression")
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
print(top_features)


####SHAP
explainer = shap.LinearExplainer(log_reg, x_train_scaled)
shap_values = explainer.shap_values(x_test_scaled)

plt.gcf().set_size_inches(14, 8)
shap.summary_plot(shap_values, x_test_scaled, feature_names=x_test.columns, max_display=15, show=False)
plt.tight_layout()
plt.show()
#print(shap_values)

# Pick an index, e.g. 0
patient_idx = 10
single_explaination = explainer(x_test_scaled[patient_idx:patient_idx+1])
plt.title('Logistice Regression SHAP')
shap.plots.waterfall(single_explaination[0])
plt.tight_layout()
