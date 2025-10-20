import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

L = 3       # Loudspeaker
J = 1024    # Filter order

dummy_input = np.array([0.2,    # Reverberation, float
                        1,      # Position,      encoded int - mid, wall, corner
                        0,      # Orientation,   degrees
                        15])    # Tilt,          degrees

dummy_q = np.zeros(L*J)


np.random.seed(0)
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 3)

# ---- 2. Split + normalisering (brug altid scaler på regression)
scaler_X = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# ---- 3. Konverter til torch-tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)



class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(input_size=X.shape[1], output_size=y.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- 6. Træningsloop
epochs = 200
batch_size = 64

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    total_loss = 0.0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices].to(device), y_train[indices].to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ---- 7. Evaluering
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()
    mse = np.mean((preds - y_test.numpy()) ** 2)
print(f"\nTest MSE: {mse:.6f}")