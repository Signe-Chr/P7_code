import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

deep_plum ="#5B3758"       # Primary – dark purple / plum
tropical_green ="#00916E"  # Secondary – teal / tropical green
rose_pink = "#DE6C83"       # Accent – rose pink
peach_orange = "#FCB97D"    # Contrast – peach / soft orange
pastel_green = "#D4E4BC"    # Contrast – pastel green
# =========================================================
# 1. Load VAST filter archive 
# =========================================================
archive_path = "PM_filter_archive.npy"

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
# 4. PCA analysis — Scree plot + choose components
# =========================================================
max_components = min(20, X_scaled.shape[1])
pca_full = PCA(n_components=max_components)
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
threshold = 0.95
# Plot scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', label='Individual variance',color=deep_plum)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's-', label='Cumulative PVE',color=tropical_green)
#plt.axhline(threshold, color=deep_plum, linestyle='--', label=f'{threshold*100}% threshold')
plt.xlabel("Number of principal components")
plt.ylabel("PVE")
plt.legend()
plt.grid(True)
plt.show()

# Choose number of components for ~90% variance

n_pca_components = np.argmax(cumulative_variance >= threshold) + 1
print(f"Number of PCA components explaining at least {threshold*100:.0f}% variance: {n_pca_components}")

# Reduce data using chosen PCA


# =========================================================
# 5. K-Medoids clustering — Silhouette-based K selection
# =========================================================
import warnings
warnings.filterwarnings("ignore", message="Cluster .* is empty!")
K_range = range(2, 10)
pcs_to_test = range(1, 9)  # 3 → 8 inclusive
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()  # flatten to easily iterate over

for idx, n_pc in enumerate(pcs_to_test):
    ax = axes[idx]

    # PCA reduction
    pca = PCA(n_components=n_pc)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"\nReduced feature matrix shape with {n_pc} PCs: {X_reduced.shape}")

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
    print(f"Optimal K for {n_pc} PCs: {best_k}")

    # Plot in subplot
    ax.plot(
        K_range,
        silhouette_scores,
        marker='o',
        color=tropical_green,
    )
    ax.set_xticks(list(K_range))
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"PCA = {n_pc} Components", color=deep_plum)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

# Hide unused subplots if fewer than 6
for ax in axes[len(pcs_to_test):]:
    ax.axis('off')

# Adjust layout
#plt.suptitle("Silhouette Scores for Different PCA Dimensions", fontsize=16, color=deep_plum)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ----------------------------------------
# 9. Final clustering with best K and NUmber of Principal Components
# ----------------------------------------
n_pc = 1
K = 2

# Step 1: PCA reduction
pca = PCA(n_components=n_pc)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced feature matrix shape with {n_pc} PCs: {X_reduced.shape}")

# Step 2: K-Medoids clustering
model = KMedoids(n_clusters=K, random_state=0)
model.fit(X_reduced)

# Step 3: Extract medoid indices and keys
medoid_indices = model.medoid_indices_
medoid_keys = [list(configurations.keys())[i] for i in medoid_indices]

print(f"\n=== Medoid keys for n_pc={n_pc}, K={K} ===")
for key in medoid_keys:
    print(" -", key)

# Step 4: Optionally, extract their q_matrices and save
medoid_dict = {key: configurations[key] for key in medoid_keys}
np.save(f"PM_medoids_PCA{n_pc}_K{K}.npy", medoid_dict)

print(f"\nSaved medoids to 'PM_medoids_PCA{n_pc}_K{K}.npy'")

if n_pc>=3:
    # ----------------------------------------
    # 9. Plot clusters using first 3 PCA components
    # ----------------------------------------
    from mpl_toolkits.mplot3d import Axes3D

    cluster_colors = [deep_plum, tropical_green, rose_pink, peach_orange, pastel_green]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(best_k):
        cluster_points = X_reduced[labels == k]
        color = cluster_colors[k % len(cluster_colors)]  # repeat if more clusters than colors
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color=color,
            label=f"Cluster {k+1}",
            alpha=0.7
        )

    # Medoids
    ax.scatter(
        X_reduced[medoid_indices, 0],
        X_reduced[medoid_indices, 1],
        X_reduced[medoid_indices, 2],
        c="black",
        marker="X",
        s=120,
        label="Medoids"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.show()