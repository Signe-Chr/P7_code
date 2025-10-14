import numpy as np
import matplotlib.pyplot as plt
deep_plum ="#5B3758"       # Primary – dark purple / plum
tropical_green ="#00916E"  # Secondary – teal / tropical green
rose_pink = "#DE6C83"       # Accent – rose pink
peach_orange = "#FCB97D"    # Contrast – peach / soft orange
pastel_green = "#D4E4BC"    # Contrast – pastel green
# =========================================================
# 1. Load VAST filter archive 
# =========================================================
archive_path = "PM_filter_archive.npy"
cluster_assignments = np.load("PM_cluster_assignments_PCA1_K2.npy", allow_pickle=True).item()

loaded = np.load(archive_path, allow_pickle=True)
if loaded.ndim == 0:
    configurations = loaded.item() 
else:
    configurations = loaded

print(f"Loaded {len(configurations)} configurations")

# =========================================================
# 2. Flatten filters for clustering
# =========================================================
Q_true = np.array([
    cfg['q_matrix'].flatten()
    for cfg in configurations.values()
])
print("True filter coefficient matrix shape:", Q_true.shape)
Q_clusters = np.load("PM_medoids_PCA2_K4.npy", allow_pickle=True).item()
print("Cluster Filter coefficient matrix shape",Q_clusters.shape)

RIR = np.array([
    cfg['RIR'].flatten()
    for cfg in configurations.values()
])

average_RIRs = {}
for cluster_id, keys in configs_per_cluster.items():
    RIRs = [configurations[k]['RIR'] for k in keys]  # adapt if your RIR key is different
    average_RIRs[cluster_id] = np.mean(RIRs, axis=0)

np.save(f"ACC_avg_RIRs_PCA{n_pc}_K{K}.npy", average_RIRs)
