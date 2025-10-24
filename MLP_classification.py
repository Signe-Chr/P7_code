import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

# global parameters
L=3
J=1024

# ============================================
# 1. Load data
# ============================================
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

X_list, filters_list = [], []

for key, inner in data.items():
    rt60 = inner.get('RT60', 0)
    phone_tilt = inner.get('Phone_tilt', 0)
    user_orient = inner.get('User_orientation', 0)
    spatial = np.array(inner.get('Spatial_position', [0, 0, 0])).ravel()

    X = np.concatenate([[rt60], [phone_tilt], [user_orient], spatial])
    X_list.append(X)
    filters_list.append(inner.get('q_matrix', np.zeros(J*L)))

# Convert to numpy arrays 
X = np.stack(X_list).astype(np.float32)  # [180,6]

# Flattened list of q_matrix arrays
filters = np.stack(filters_list).astype(np.float32)
filters = filters.reshape(filters.shape[0], -1)  # (180,3072)

print(f"Input features shape: {X.shape}")
print(f"Filter dictionary shape (flattened): {filters.shape}")

# ============================================
# 2. Convert to torch tensors
# ============================================
configs_tensor = torch.from_numpy(X)            # [180, 6]
filters_tensor = torch.from_numpy(filters)      # [180, 3072]

input_size = configs_tensor.shape[1]
num_filters = filters_tensor.shape[0]
filter_dim = filters_tensor.shape[1]

# ============================================
# 3. Model definition
# ============================================
class SoftFilterNet(nn.Module):
    def __init__(self, input_size, num_filters, filter_dim, filters_tensor):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_filters)
        self.register_buffer("filters", filters_tensor)  # fixed filters (not trainable)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        weights = F.softmax(logits, dim=1)            # [batch, num_filters]
        combined = torch.matmul(weights, self.filters) # [batch, filter_dim]
        return combined, weights

# Instantiate model
model = SoftFilterNet(input_size, num_filters, filter_dim, filters_tensor)
summary(model, input_size=(input_size,))

# ============================================
# 4. Training setup
# ============================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================
# 5. Training loop
# ============================================
model.train()
for epoch in range(100):
    outputs, weights = model(configs_tensor)            # [N, filter_dim]
    loss = criterion(outputs, filters_tensor)           # [N, filter_dim] vs [N, filter_dim]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
        print(f"Epoch [{epoch+1}/100]  MSE: {loss.item():.6e}  Avg Entropy: {entropy:.3f}")

# ============================================
# 6. Save model weights
# ============================================
torch.save(model.state_dict(), "mlp_weights.pth")
print("Model weights saved to mlp_weights.pth")

# ============================================
# 7. Evaluation on a new configuration
# ============================================
model.eval()
test_config = torch.FloatTensor([[0.27, 3.14, 1.57, 5, 5, 1.7]])  # adjust to your feature count!

with torch.no_grad():
    predicted_filter, weights = model(test_config)
    probs = weights.squeeze().cpu().numpy()

# ============================================
# 8. Display results
# ============================================
print("\n=== Test Prediction ===")
print("Input config:", test_config.squeeze().numpy())
print("Softmax probabilities (top 5):", np.sort(probs)[-5:])
print("Sum of probabilities:", probs.sum())
print("Predicted blended filter shape:", predicted_filter.shape)

# Inspect top-3 contributing filters
top3_idx = np.argsort(probs)[-3:][::-1]
print("Top 3 contributing filter indices:", top3_idx)
print("Top 3 weights:", probs[top3_idx])

predicted_filter_np = predicted_filter.squeeze().cpu().numpy()
print(f"Predicted blended filter length: {len(predicted_filter_np)}")


# ============================================
# 9. Plot MSE
# ============================================
model.eval()

# Get full outputs and weights for all training configs
with torch.no_grad():
    full_pred, full_weights = model(configs_tensor)  # [N, F], [N, num_filters]

# Convert to CPU numpy
full_weights_np = full_weights.cpu().numpy()  # [N, num_filters]
filters_np = filters_tensor.cpu().numpy()     # [N, F]
filter_dict = model.filters.cpu().numpy()     # [num_filters, F]
N, num_filters = full_weights_np.shape

mse_vs_k = []

# Evaluate truncated reconstructions for increasing k
for k in range(1, num_filters + 1):
    total_mse = 0.0
    for i in range(N):
        w = full_weights_np[i]
        # indices of top-k filters
        topk_idx = np.argsort(w)[-k:]
        topk_w = w[topk_idx]
        topk_w /= topk_w.sum()  # renormalize to sum to 1
        # recompute blended filter using only top-k
        blended = np.dot(topk_w, filter_dict[topk_idx])
        # MSE with ground-truth
        mse = np.mean((blended - filters_np[i]) ** 2)
        total_mse += mse
    avg_mse = total_mse / N
    mse_vs_k.append(avg_mse)

# Plot MSE vs number of filters used
plt.figure(figsize=(7,5))
plt.plot(range(1, num_filters + 1), mse_vs_k, marker='o')
plt.xlabel("Number of filters in linear combination (k)")
plt.ylabel("Average MSE")
plt.title("Reconstruction error vs. number of contributing filters")
plt.grid(True)
plt.tight_layout()
plt.show()
