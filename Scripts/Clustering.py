import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# 1. Load your data
df = pd.read_csv('/Users/alishasarkar/Documents/Python Lab/Diabetes_new/Diabetes-Readmission-Predictor/Alisha_Approach/Data/clean/Clean_data_for_train(1).csv')

# 2. Select relevant features
features = ['num_lab_procedures', 'num_procedures', 'num_medications', 'max_glu_serum', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']  
X = df[features]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_jitter = X_scaled + np.random.normal(0, 0.01, X_scaled.shape)

# 4. Find optimal k using silhouette score
sil_scores = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = K_range[sil_scores.index(max(sil_scores))]
print(f"Best k: {best_k}")

# 5. Fit KMeans with best k
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# 6. UMAP for 2D and 3D
reducer_2d = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer_2d.fit_transform(X_scaled_jitter)

reducer_3d = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer_3d.fit_transform(X_scaled_jitter)

# 7. Plot 2D UMAP
plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='Set1', s=40)
plt.title("Patient Subtype Clusters (2D UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()

# 8. Plot 3D UMAP
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], c=clusters, cmap='Set1', s=50)
ax.set_title('Patient Subtype Clusters (3D UMAP)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
fig.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# 9. Analyze cluster characteristics
print(df.groupby('cluster')[features].mean())