import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# 1. Load your data (replace with your dataframe)
df = pd.read_csv('Data/Clean_data_for_train(1).csv')

# 2. Select relevant features (example)
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

# 6. UMAP for 2D visualization
reducer = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer.fit_transform(X_scaled_jitter)

# # 6. PCA for 2D visualization
# pca = PCA(n_components=2)
# embedding = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embedding_3d[:,0], embedding_3d[:,1], embedding_3d[:,2], c=clusters, cmap='tab10', s=50)
ax.set_title('Patient Subtype Clusters (3D UMAP)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
fig.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# 7. Analyze cluster characteristics
print(df.groupby('cluster')[features].mean())