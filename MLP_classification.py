import torch
import numpy as np
from torchsummary import summary

# ---- 1. Load data from VAST archive
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

X_list, filters_list = [], []

for key, inner in data.items():
    # Robust handling of features (fallbacks if missing)
    rt60 = inner.get('RT60', 0)                         # Reverberation time
    phone_tilt = inner.get('Phone_tilt', 0)             # In radians
    user_orient = inner.get('User_orientation', 0)      # In radians
    spatial = inner.get('Spatial_position', [0, 0, 0])  # (x, y, z)
    spatial = np.array(spatial).ravel()

    # Input feature vector
    X = np.concatenate([
        [rt60],
        [phone_tilt],
        [user_orient],
        spatial
    ])

    # Store
    X_list.append(X)
    filters_list.append(inner.get('q_matrix', np.zeros(3072)))  # assume L=3, J=1024

# Convert to numpy arrays
X = np.stack(X_list).astype(np.float32)
filters = np.stack(filters_list).astype(np.float32)

# Create labels (each q_matrix = one class)
labels = np.arange(len(X), dtype=np.int64)

# Convert to torch tensors
configs_tensor = torch.from_numpy(X)
labels_tensor = torch.from_numpy(labels)

# ---- 2. Model definition
num_classes = len(X)  # one per filter
input_size = X.shape[1]

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, num_classes)
)
summary(model, input_size=(input_size,))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- 3. Training
model.train()
for epoch in range(100):
    outputs = model(configs_tensor)
    loss = criterion(outputs, labels_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels_tensor).float().mean()
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}, Acc: {accuracy.item():.3f}")

# ---- 4. Evaluation
model.eval()
test_config = torch.FloatTensor([[0.1, 1, 45, 5, 5, 1.7]]) # 6 features: rt60, tilt, orient, x, y, z

with torch.no_grad():
    predictions = model(test_config)
    predicted_class = torch.argmax(predictions, dim=1).item()
    probabilities = torch.softmax(predictions, dim=1).squeeze().cpu().numpy()

print(f"\n=== Test Prediction ===")
print(f"Input config: {test_config.squeeze().numpy()}")
print(f"Predicted class index: {predicted_class}")
print(f"Top 5 probabilities: {np.sort(probabilities)[-5:]}")

# ---- 5. Retrieve corresponding filter
selected_filter = filters_list[predicted_class]
print(f"Selected filter length: {len(selected_filter)}")
