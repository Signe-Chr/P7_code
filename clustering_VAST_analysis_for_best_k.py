import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# =========================================================
# 1. Load VAST filter archive 
# =========================================================
archive_path = "VAST_filter_archive.npy"

loaded = np.load(archive_path, allow_pickle=True)
if loaded.ndim == 0:
    configurations = loaded.item()
else:
    configurations = loaded

print(f"Loaded {len(configurations)} configurations")

# =========================================================
# 2. Flatten filters for clustering
# =========================================================
X = np.array([
    cfg['q_matrix'].flatten()
    for cfg in configurations.values()
])
print("Feature matrix shape:", X.shape)

# =========================================================
# 3. Standardize features
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# 4. PCA analysis — Scree plot + choose components
# =========================================================
max_components = min(50, X_scaled.shape[1])
pca_full = PCA(n_components=max_components)
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
threshold = 0.95
# Plot scree plot
plt.figure(figsize=(8, 5))
#plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', label='Individual variance')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's-', label='Cumulative variance')
plt.axhline(threshold, color='r', linestyle='--', label=f'{threshold*100}% threshold')
plt.xlabel("Number of PCA components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot — Filter Coefficients")
plt.legend()
plt.grid(True)
plt.show()

# Choose number of components for ~90% variance

n_pca_components = np.argmax(cumulative_variance >= threshold) + 1
print(f"Number of PCA components explaining at least {threshold*100:.0f}% variance: {n_pca_components}")

# Reduce data using chosen PCA
pca = PCA(n_components=n_pca_components)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced feature matrix shape: {X_reduced.shape}")

# =========================================================
# 5. K-Medoids clustering — Silhouette-based K selection
# =========================================================
import warnings
warnings.filterwarnings("ignore", message="Cluster .* is empty!")
K_range = range(2, 8)
silhouette_scores = []

for k in K_range:
    model = KMedoids(n_clusters=k, random_state=0)
    try:
        model.fit(X_reduced)
        labels = model.labels_
        if len(np.unique(labels)) < 2:
            print(f"K={k} skipped: only {len(np.unique(labels))} cluster(s)")
            silhouette_scores.append(-1)
            continue
        score = silhouette_score(X_reduced, labels)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score:.3f}")
    except Exception as e:
        print(f"K={k} skipped due to error: {e}")
        silhouette_scores.append(-1)

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K by Silhouette Score: {best_k}")

# =========================================================
# 6. Final clustering with best K
# =========================================================
best_model = KMedoids(n_clusters=best_k, random_state=0)
best_model.fit(X_reduced)
labels = best_model.labels_
medoid_indices = best_model.medoid_indices_

unique_labels, counts = np.unique(labels, return_counts=True)
print("\nNumber of points in each cluster:")
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")
# =========================================================
# 7. Extract medoid filters (original, not PCA)
# =========================================================
medoid_keys = [list(configurations.keys())[i] for i in medoid_indices]
medoid_filters = [configurations[key]['q_matrix'] for key in medoid_keys]

print("\nMedoid configurations (representative filters):")

for key in medoid_keys:
    print(" -", key)
medoid_dict = {
    key: configurations[key] for key in medoid_keys
}
np.save("VAST_medoids_full.npy", medoid_dict)
# =========================================================
# 8. Plot Silhouette Scores vs K
# =========================================================
plt.figure(figsize=(6, 4))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xticks(K_range)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K — K-Medoids Clustering")
plt.grid(True)
plt.show()

# ----------------------------------------
# 9. Plot clusters using first 3 PCA components
# ----------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
for k in range(best_k):
    cluster_points = X_reduced[labels==k]
    ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], label=f"Cluster {k+1}", alpha=0.7)

ax.scatter(
    X_reduced[medoid_indices,0],
    X_reduced[medoid_indices,1],
    X_reduced[medoid_indices,2],
    c="black", marker="X", s=120, label="Medoids"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()