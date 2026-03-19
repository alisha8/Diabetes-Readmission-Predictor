import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/clean/Clean_data_for_gui.csv")

# Count Plot — Age vs. Readmission
sns.countplot(x='age', hue='readmitted', data=df)
plt.title("Age Group vs Readmission (<30 days)")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Race/Gender vs. Readmission
# Race
sns.countplot(x='race', hue='readmitted', data=df)
plt.title("Race vs Readmission")
plt.xlabel("Race")
plt.ylabel("Count")
plt.show()

# # Gender
sns.countplot(x='gender', hue='readmitted', data=df)
plt.title("Gender vs Readmission")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Medication Change vs. Readmission
sns.countplot(x='change', hue='readmitted', data=df)
plt.title("Medication Change vs Readmission")
plt.xlabel("Change in Medication")
plt.ylabel("Count")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12,12))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# # PCA 2D
# features = df.drop(['readmitted'], axis=1).select_dtypes(include=np.number)
# X_scaled = StandardScaler().fit_transform(features)

# pca = PCA(n_components=2)
# components = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df['readmitted'])
# plt.title("PCA Projection")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# # UMAP
# reducer = umap.UMAP()
# features = df.drop(['readmitted'], axis=1).select_dtypes(include=np.number)

# # Scale data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(features)

# # Initialize UMAP reducer
# reducer = umap.UMAP()

# # Fit and transform scaled data
# embedding = reducer.fit_transform(X_scaled)

# print(embedding.shape)

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['readmitted'])
# plt.title("UMAP Projection")
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.show()


# # Random Forest Classifier
# X = df.drop(['readmitted'], axis=1).select_dtypes(include=np.number)
# y = df['readmitted']

# model = RandomForestClassifier()
# model.fit(X, y)

# importances = model.feature_importances_
# features = X.columns
# imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# sns.barplot(x='Importance', y='Feature', data=imp_df.head(10))
# plt.title("Top 10 Feature Importances")
# plt.tight_layout()
# plt.show()