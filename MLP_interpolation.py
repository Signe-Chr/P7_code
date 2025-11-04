import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(69420)
torch.manual_seed(69420)

# ---- 1. Load data
def load_data(dataset="VAST_filter_archive.npy"):
    data = np.load(dataset, allow_pickle=True).item()

    X_list, filters_list, rt60_list = [], [], []

    for key, inner in data.items():
        rt60 = inner.get('RT60', 0.0)
        phone_tilt = inner.get('Phone_tilt', 0.0)
        user_orient = inner.get('User_orientation', 0.0)
        spatial = np.array(inner.get('Spatial_position', [0, 0, 0]), dtype=np.float32).ravel()

        X = np.concatenate([[rt60], [phone_tilt], [user_orient], spatial])
        X_list.append(X)
        filters_list.append(inner.get('q_matrix', np.zeros(3072, dtype=np.float32)))
        rt60_list.append(rt60)

    # Convert to arrays
    X = np.stack(X_list).astype(np.float32)
    filters = np.stack(filters_list).astype(np.float32)
    rt60_array = np.array(rt60_list)

    num_samples, input_size = X.shape
    filter_dim = filters.shape[1]

    print(f"Total samples: {num_samples}, Feature size: {input_size}, Filter length: {filter_dim}")

    # ---- 2. Train/test split by RT60
    unique_rooms = np.unique(rt60_array)
    print("Unique RT60 values:", unique_rooms)

    test_room = unique_rooms[0]  # Pick one RT60 as unseen test room
    train_mask = rt60_array != test_room
    test_mask = rt60_array == test_room

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = filters[train_mask], filters[test_mask]

    # Flatten filters (3×1024 → 3072)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples (RT60={test_room})")

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    # Create unique 2D filter dictionary
    filters_flat = torch.from_numpy(filters).view(filters.shape[0], -1)
    filters_tensor = torch.unique(filters_flat, dim=0)
    num_filters, filter_dim = filters_tensor.shape

    print(f"Filter dictionary shape: {filters_tensor.shape}")
    return X_train, X_test, y_train, y_test, X_train_t, y_train_t, X_test_t, y_test_t, num_filters, filter_dim, filters_tensor, input_size

# ---- 3. Model
class SoftFilterNet(nn.Module):
    def __init__(self, input_size, num_filters, filter_dim, filters_tensor):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, num_filters)
        self.register_buffer("filters", filters_tensor)  # [num_filters, filter_dim]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)                        # [batch, num_filters]
        weights = F.softmax(logits, dim=1)          # [batch, num_filters]
        combined = weights @ self.filters           # [batch, filter_dim]
        return combined, weights


if __name__== "__main__":
# ---- 4. Training loop
    X_train, X_test, y_train, y_test, X_train_t, y_train_t, X_test_t, y_test_t, num_filters, filter_dim, filters_tensor, input_size = load_data()
    model = SoftFilterNet(input_size, num_filters, filter_dim, filters_tensor)
    summary(model, input_size=(input_size,))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for epoch in range(200):
        predicted_filters, weights = model(X_train_t)  # two outputs
        loss = criterion(predicted_filters, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "mlp_weights.pth")
    print("Model weights saved to mlp_weights.pth")
    torch.save(filters_tensor, "filters_tensor.pt")
    print("Filters saved to filters_tensor.pt")


           # ---- Test model output on specific input
    model.eval()
    with torch.no_grad():
        test_input = np.concatenate([[0.6], [np.deg2rad(15)], [np.pi/2], [2, 2, 4]]).astype(np.float32)
        test_input_t = torch.from_numpy(test_input).unsqueeze(0)  # [1, input_size]

        pred_filter, pred_weights = model(test_input_t)

        print(f"Predicted filter shape: {pred_filter.shape}")
        print(f"Weights shape:          {pred_weights.shape}")

        # Save all predicted filter coefficients to text file
        np.savetxt("predicted_filter_1.txt", pred_filter.squeeze().cpu().numpy(), fmt="%.8f")
        print(f"\nAll {pred_filter.numel()} predicted filter coefficients saved to 'predicted_filter.txt'")

        # Optional: preview first few coefficients
        print("\nFirst 10 predicted filter values:")
        print(pred_filter[0, :10].cpu().numpy())

    with torch.no_grad():
        _, weights = model(test_input_t)  # weights shape: [1, num_filters]

        # ---- Select filter with highest softmax probability
        max_idx = torch.argmax(weights, dim=1).item()
        selected_filter = filters_tensor[max_idx]
        

        print(f"Selected filter index: {max_idx}")
        print(f"Highest softmax probability: {weights[0, max_idx].item():.6f}")

        # ---- Save the selected filter coefficients
        np.savetxt("predicted_filter_top1.txt", selected_filter.cpu().numpy(), fmt="%.8f")
        print(f"Selected filter (index {max_idx}) saved to 'predicted_filter_top1.txt'")

        # ---- Optional: show first few coefficients
        print("\nFirst 10 coefficients of selected filter:")
        print(selected_filter[:10].cpu().numpy())

"""
# ---- 5. Evaluation with top-k filter combination
model.eval()
num_filters = filters_tensor.shape[0]
avg_mse_list = []

with torch.no_grad():
    # ---- Compute training loss
    train_predicted, _ = model(X_train_t)
    train_loss = torch.mean((train_predicted - y_train_t) ** 2)
    print(f"Training Loss: {train_loss.item():.6f}")

    # ---- Compute test loss
    test_predicted, test_weights = model(X_test_t)
    test_loss = torch.mean((test_predicted - y_test_t) ** 2)
    print(f"Test Loss: {test_loss.item():.6f}")

    # ---- Top-k analysis
    for k in range(1, num_filters + 1):
        mse_sum = 0.0
        for i in range(X_test_t.shape[0]):
            w = test_weights[i].cpu().numpy()
            # top-k indices
            topk_idx = np.argsort(w)[-k:][::-1]
            topk_w = np.zeros_like(w)
            topk_w[topk_idx] = w[topk_idx]
            topk_w /= topk_w.sum()  # renormalize

            # compute weighted filter
            pred_filter = torch.from_numpy(topk_w).unsqueeze(0) @ filters_tensor
            mse = torch.mean((pred_filter - y_test_t[i].unsqueeze(0)) ** 2).item()
            mse_sum += mse

        avg_mse_list.append(mse_sum / X_test_t.shape[0])

# ---- 6. Find best k
avg_mse_array = np.array(avg_mse_list)
best_idx = np.argmin(avg_mse_array)
best_k = best_idx + 1
print(f"\nBest MSE = {avg_mse_array[best_idx]:.6f} occurs at k = {best_k} filters")

# ---- 7. Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_filters + 1), avg_mse_list, marker='o')
plt.xlabel("Number of top-k filters used")
plt.ylabel("Average MSE on test set")
plt.title("Effect of number of filters on reconstruction error")
plt.grid(True)
plt.show()
"""