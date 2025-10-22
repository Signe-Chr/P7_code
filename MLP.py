import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

L = 3       # Loudspeaker
J = 1024    # Filter order

dummy_input = np.array([0.2,    # Reverberation, float
                        1,      # Position,      encoded int - mid, wall, corner
                        0,      # Orientation,   degrees
                        15])    # Tilt,          degrees

dummy_q = np.zeros(L*J)

# ---- 1. Load data
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

X_list, y_list = [], []

for key, inner in data.items():
    # Input: flatten source positions (3x3 -> 9)
    X = np.ravel(inner['sources_position'])

    # Output: flatten q_matrix (3x1048 -> 3144)
    y = np.ravel(inner['q_matrix'])

    X_list.append(X)
    y_list.append(y)

X = np.stack(X_list)
y = np.stack(y_list)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---- 2. Train/test split and scaling
scaler_X = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---- 3. Define the network
class FilterNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FilterNet(input_size=X.shape[1], output_size=y.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- 4. Training loop
epochs = 200
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    total_loss = 0.0

    for i in range(0, X_train.size(0), batch_size):
        idx = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[idx].to(device), y_train[idx].to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ---- 5. Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()
    mse = np.mean((preds - y_test.numpy()) ** 2)
print(f"\nTest MSE: {mse:.6f}")
