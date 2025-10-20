import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

# Colors (for future plotting)
deep_plum ="#5B3758"
tropical_green ="#00916E"
rose_pink = "#DE6C83"
peach_orange = "#FCB97D"
pastel_green = "#D4E4BC"

# =========================================================
# 1. Load data
# =========================================================
archive_path = "PM_filter_archive.npy"
cluster_assignments = np.load("PM_cluster_assignments_PCA1_K2.npy", allow_pickle=True).item()

loaded = np.load(archive_path, allow_pickle=True)
configurations = loaded.item() if loaded.ndim == 0 else loaded
print(f"Loaded {len(configurations)} configurations")

# =========================================================
# 2. Load medoids (cluster filters)
# =========================================================
Q_clusters = np.load("PM_medoids_PCA2_K4.npy", allow_pickle=True).item()
print("Loaded", len(Q_clusters), "cluster medoids")

# =========================================================
# 3. Extract filter and RIR arrays
# =========================================================
Q_true = np.array([cfg['q_matrix'].flatten() for cfg in configurations.values()])
print("True filter coefficient matrix shape:", Q_true.shape)

RIR = np.array([cfg['RIR'].flatten() for cfg in configurations.values()])
print("RIR matrix shape:", RIR.shape)

# =========================================================
# 4. Build cluster grouping from assignments
# =========================================================
configs_per_cluster = defaultdict(list)
for key, cluster_id in cluster_assignments.items():
    configs_per_cluster[cluster_id].append(key)

# =========================================================
# 5. Compute average RIR per cluster
# =========================================================
average_RIRs = {}
for cluster_id, keys in configs_per_cluster.items():
    RIRs = [configurations[k]['RIR'] for k in keys]
    average_RIRs[cluster_id] = np.mean(RIRs, axis=0)
    print(f"Cluster {cluster_id}: {len(RIRs)} configs → avg RIR length = {len(average_RIRs[cluster_id])}")

# Optional normalization
for cluster_id in average_RIRs:
    avg_rir = average_RIRs[cluster_id]
    average_RIRs[cluster_id] = avg_rir / np.linalg.norm(avg_rir)

# =========================================================
# 6. Save average RIRs for later NN use
# =========================================================
n_pc = 1
K = 2
np.save(f"PM_avg_RIRs_PCA{n_pc}_K{K}.npy", average_RIRs)
print(f"\nSaved average RIRs to 'PM_avg_RIRs_PCA{n_pc}_K{K}.npy'")
# =========================================================
# 7. Create data set for traning and test
# =========================================================
inputs = []
targets = []
for key, cfg in tqdm(configurations.items(), desc="Building dataset"):
    # Step 1: Get true filter and RIR
    q_true = cfg["q_matrix"].flatten()
    rir = cfg["RIR"].flatten()

    # Step 2: Get cluster assignment
    cluster_id = cluster_assignments[key]

    # Step 3: Get cluster representative (medoid filter)
    q_cluster = Q_clusters[cluster_id]["q_matrix"].flatten()

    # Step 4: Compute correction
    delta_q = q_true - q_cluster

    # Store pair
    inputs.append(rir)
    targets.append(delta_q)

# Convert to arrays
X = np.stack(inputs)
Y = np.stack(targets)

print("Dataset shapes:")
print("  X (RIRs):", X.shape)
print("  Y (Δq):", Y.shape)

np.savez("NN_dataset_RIR_to_DeltaQ.npz", X=X, Y=Y)
print("\n Saved dataset to 'NN_dataset_RIR_to_DeltaQ.npz'")
# =========================================================
# 8. Train Neural Network
# =========================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
import tensorflow as tf
from tensorflow.keras import layers, models

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(output_dim)  # linear output for regression
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

model.save("NN_RIR_to_DeltaQ.h5")
print("\n Model saved as 'NN_RIR_to_DeltaQ.h5'")
