import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
np.random.seed(69420)
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

filters_train_dict = y_train
filters_test_true = y_test

print("Training samples:", X_train.shape[0])
print("Test samples (unseen room):", X_test.shape[0])

# ---- 4. Define model
model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, 512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512, y_train.shape[0])
                    )
summary(model, input_size=(input_size,))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- 5. Training loop
model.train()
for epoch in range(200):
    logits = model(X_train)                 # [N_train, N_total]
    weights = F.softmax(logits, dim=1)      # softmax weights over all filters
    predicted_filters = weights @ y_train  # [N_train, filter_length]

    loss = criterion(predicted_filters, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.6f}")

# ---- 6. Evaluation with top-k filter combination
model.eval()
num_filters = y_train.shape[0]  # total filters
avg_mse_list = []

with torch.no_grad():
    # Compute logits and softmax for all test configs
    test_logits = model(X_test)               # [N_test, N_total]
    test_weights = F.softmax(test_logits, dim=1)  # [N_test, N_total]
    train_logits = model(X_train)                 # [N_train, N_total]
    train_weights = torch.softmax(train_logits, dim=1)  # softmax over filters
    train_predicted = train_weights @ y_train      # weighted sum
    train_loss = torch.mean((train_predicted - y_train)**2)  # MSE
    print(f"Training Loss: {train_loss.item():.6f}")
    
    test_logits = model(X_test)                  # [N_test, N_total]
    test_weights = torch.softmax(test_logits, dim=1)
    test_predicted = test_weights @ y_train
    test_loss = torch.mean((test_predicted - y_test)**2)
    print(f"Test Loss: {test_loss.item():.6f}")
    for k in range(1, num_filters+1):
        mse_sum = 0.0
        for i in range(X_test.shape[0]):
            w = test_weights[i].cpu().numpy()
            # top-k indices
            topk_idx = np.argsort(w)[-k:][::-1]
            topk_w = np.zeros_like(w)
            topk_w[topk_idx] = w[topk_idx]
            topk_w /= topk_w.sum()  # renormalize

            # compute weighted filter
            pred_filter = torch.from_numpy(topk_w).unsqueeze(0) @ y_train  # [1, filter_length]
            mse = torch.mean((pred_filter - y_test[i].unsqueeze(0))**2).item()
            mse_sum += mse

        avg_mse_list.append(mse_sum / X_test.shape[0])
        
avg_mse_array = np.array(avg_mse_list)
# Find index of minimum MSE
best_idx = np.argmin(avg_mse_array)  # index in 0-based Python
best_k = best_idx + 1                 # k = 1-based

print(f"\nBest MSE = {avg_mse_array[best_idx]:.6f} occurs at k = {best_k} filters")
    
# ---- 7. Plot average MSE vs top-k filters
plt.figure(figsize=(8,5))
plt.plot(range(1, num_filters+1), avg_mse_list, marker='o')
plt.xlabel("Number of top-k filters used")
plt.ylabel("Average MSE on test set")
plt.title("Effect of number of filters on reconstruction error")
plt.grid(True)
plt.show()

cum_prob_list = []

# Compute cumulative probabilities for each test sample
for i in range(X_test.shape[0]):
    w = test_weights[i].cpu().numpy()
    w_sorted = np.sort(w)[::-1]          # descending
    cum_prob = np.cumsum(w_sorted)       # cumulative sum
    cum_prob_list.append(cum_prob)

cum_prob_array = np.stack(cum_prob_list)      # [N_test, N_total]

# Average cumulative probability across all test samples
avg_cum_prob = np.mean(cum_prob_array, axis=0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, y_train.shape[0] + 1), avg_cum_prob, marker='o')
plt.xlabel("Top-k filters")
plt.ylabel("Average cumulative softmax probability")
plt.title("Average cumulative softmax probability vs top-k filters")
plt.grid(True)
plt.show()