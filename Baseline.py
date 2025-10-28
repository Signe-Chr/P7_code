import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(69420)
import time
# ---- 1. Load data from VAST archive
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()
X_list, filters_list,rt_60_list = [], [],[]

for key, inner in data.items():
    # Robust handling of features
    rt60 = inner.get('RT60', 0.0)
    phone_tilt = inner.get('Phone_tilt', 0.0)
    user_orient = inner.get('User_orientation', 0.0)
    spatial = np.array(inner.get('Spatial_position', [0, 0, 0]), dtype=np.float32).ravel()

    # Input feature vector
    X = np.concatenate([[rt60], [phone_tilt], [user_orient], spatial])
    X_list.append(X)

    # q_matrix = target filter coefficients
    q = inner.get('q_matrix', np.zeros(3072, dtype=np.float32))
    filters_list.append(q.flatten())
    
    rt_60_list.append(rt60)

# ---- 2. Prepare arrays and tensors
X = np.stack(X_list).astype(np.float32)        # [N_total, num_features]
filters = np.stack(filters_list).astype(np.float32)  # [N_total, filter_length]
rt60_array=np.array(rt_60_list)

num_total, input_size = X.shape
filter_length = filters.shape[1]

configs_tensor = torch.from_numpy(X)        # [N_total, num_features]
filters_tensor = torch.from_numpy(filters)  # [N_total, filter_length]

print(f"Configs shape: {configs_tensor.shape}")
print(f"Filters shape: {filters_tensor.shape}")

# ---- 3. Split into train/test sets
unique_rooms = np.unique(rt60_array)
print("All rooms (RT60 values):", unique_rooms)

# Choose a test room
test_room = unique_rooms[0]  # pick first room as test
train_mask = rt60_array != test_room
test_mask = rt60_array == test_room

# Apply masks
X_train = configs_tensor[train_mask]
X_test = configs_tensor[test_mask]
y_train = filters_tensor[train_mask]
y_test = filters_tensor[test_mask]

print("Training samples:", X_train.shape[0])
print("Test samples (unseen room):", X_test.shape[0])

#Baseline - Compute all MSEs between the true value, and the filters in the dictionary, and chosse the smallest one. The Baseline mse is found as an average across the entire test set
# [N_test, N_train, filter_length]
t0 = time.time()
diffs = y_test.unsqueeze(1) - y_train.unsqueeze(0)
mse_matrix = torch.mean(diffs ** 2, dim=2)  # [N_test, N_train]
best_mse_per_test = torch.min(mse_matrix, dim=1).values
baseline_mse = best_mse_per_test.mean().item()
t1 = time.time()
print(f"Baseline time per query: {(t1 - t0)/len(y_test):.6f} s")
print(f"Exhaustive search baseline MSE: {baseline_mse:.6f}")
t2 = time.time()
# -----------------------------
# 3. Cosine similarity baseline
# -----------------------------
# Normalize filters to unit norm

y_train_norm = F.normalize(y_train, p=2, dim=1)
y_test_norm = F.normalize(y_test, p=2, dim=1)

# Compute cosine similarity between all test and train filters
# Result shape: [N_test, N_train]
similarity_matrix = torch.mm(y_test_norm, y_train_norm.T)

# For each test sample, pick the most similar training filter
best_similarities, best_indices = torch.max(similarity_matrix, dim=1)

# Convert to cosine distance (1 - similarity)
cosine_distances = 1 - best_similarities
mean_cosine_distance = cosine_distances.mean().item()
t3 = time.time()
print(f"Exhaustive search baseline (cosine distance): {mean_cosine_distance:.6f}")
print(f"Baseline time per query cosine similarity: {(t3 - t2)/len(y_test):.6f} s")