import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

deep_plum ="#5B3758"       # Primary – dark purple / plum
tropical_green ="#00916E"  # Secondary – teal / tropical green
rose_pink = "#DE6C83"       # Accent – rose pink
peach_orange = "#FCB97D"    # Contrast – peach / soft orange
pastel_green = "#D4E4BC"    # Contrast – pastel green
# =========================================================
# 1. Load VAST filter archive 
# =========================================================
archive_path = "ACC_filter_archive.npy"

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
print("Filter coefficient matrix shape:", X.shape)

# =========================================================
# 3. Standardize features
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# =========================================================
# 2. Compute silhouette scores for different K values
# =========================================================
K_range = range(2, 51)
silhouette_scores = []
inertia=[]

for K in K_range:
    print(f"\nFitting KMeans with K={K} ...")
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=20)
    kmeans.fit(X_scaled)
    
    labels = kmeans.labels_
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    inertia.append(kmeans.inertia_)
    
    # Count points in each cluster
    cluster_sizes = np.bincount(labels)
    
    print(f"Silhouette Score for K={K}: {score:.4f}")
    print(f"Cluster sizes: {cluster_sizes}")
    
fig, axes = plt.subplots(2, 1, figsize=(14, 5))  # 1 row, 2 columns

# -----------------------------
# Left plot: Elbow / Inertia
# -----------------------------
axes[0].plot(K_range, inertia, 'o-', linewidth=2, markersize=6, color="#5B3758")
axes[0].set_xticks(K_range)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
axes[0].grid(True, linestyle='--', alpha=0.6)

# -----------------------------
# Right plot: Silhouette Scores
# -----------------------------
axes[1].plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=6, color="#00916E")
axes[1].set_xticks(K_range)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# =========================================================
# 4. Report the best K
# =========================================================
#best_K = K_range[np.argmax(silhouette_scores)]
#print(f"\nOptimal K based on silhouette score: {best_K}")
best_K=6
final_model=KMeans(n_clusters=best_K, random_state=0, n_init=20)
final_model.fit(X_scaled)

label=final_model.labels_
centroids_scaled=final_model.cluster_centers_
print("Centroids (standardized space):", centroids_scaled.shape)

centroids_original = scaler.inverse_transform(centroids_scaled)
print("Centroids (original feature space):", centroids_original.shape)
np.save(f"ACC_kmeans_centroids_scaled_K{best_K}.npy", centroids_scaled)
np.save(f"ACC_kmeans_centroids_original_K{best_K}.npy", centroids_original)

print(f"\n Saved centroids to:")
print(f" - 'ACC_kmeans_centroids_scaled_K{best_K}.npy'  (standardized space)")
print(f" - 'ACC_kmeans_centroids_original_K{best_K}.npy' (original feature space)")

# =========================================================
# 1. Reduce data to 3D PCA space for visualization only
# =========================================================
pca_vis = PCA(n_components=3, random_state=0)
X_vis = pca_vis.fit_transform(X_scaled)

# Project centroids into the same PCA space
centroids_vis = pca_vis.transform(centroids_scaled)

# =========================================================
# 2. 3D scatter plot of clusters
# =========================================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Optional: custom colors
cluster_colors = ["#5B3758", "#00916E", "#DE6C83", "#FCB97D", "#D4E4BC"]

for k in range(K):
    points = X_vis[labels == k]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=cluster_colors[k % len(cluster_colors)],
        label=f"Cluster {k+1}",
        alpha=0.7,
        s=30
    )

# Plot centroids as large black X’s
ax.scatter(
    centroids_vis[:, 0],
    centroids_vis[:, 1],
    centroids_vis[:, 2],
    c='black',
    marker='X',
    s=200,
    label='Centroids'
)

ax.set_title(f"K-Means Clusters (visualized in 3D PCA space, K={K})", color="#333")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.show()

