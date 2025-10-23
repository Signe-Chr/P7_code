import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary

L = 3       # Loudspeaker
J = 1024    # Filter order

dummy_input = np.array([2.2,    # Reverberation, float
                        0.78,      # Phone tilt, degrees
                        3.14,      # Orientation,  degrees
                        5, 10, 1.7])    # Spatial position, (x, y, z)       

dumm = torch.tensor(dummy_input, dtype=torch.float32)
print(dumm, dummy_input)

# ---- 1. Load data
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()
for key, inner in data.items():
    print(f"--- Key: {key} ---")
    for field in inner:
        value = inner[field]
        if isinstance(value, np.ndarray):
            print(f"{field}: numpy array, shape = {value.shape}")
        elif isinstance(value, list):
            print(f"{field}: list, length = {len(value)}")
        else:
            print(f"{field}: type = {type(value)}, value = {value}")
    break
X_list, y_list = [], []


for key, inner in data.items():
    # Robust håndtering af input features (fallback til 0 hvis mangler)
    rt60 = inner.get('RT60', 0)                         # 2.5
    phone_tilt = inner.get('Phone_tilt', 0)             # I radianer: 0.261, 0.785, 1.309
    user_orient = inner.get('User_orientation', 0)      # I radianer: 0, 1.57, 3.14, 4.71
    spatial = inner.get('Spatial_position', [0,0,0])    # (x, y, z): (5, 5 ,1.7) betyder i midten af rummet og i højde 1.7m
    spatial = np.array(spatial).ravel()                 # flad ud til 1D
    
    X = np.concatenate([
        [rt60],
        [phone_tilt],
        [user_orient],
        spatial
    ])
    
    # Output: flatten q_matrix
    y = np.ravel(inner.get('q_matrix', np.zeros(L*J)))
    
    X_list.append(X)
    y_list.append(y)

X = np.stack(X_list)
y = np.stack(y_list)

#print(X)
#print("X shape:", X.shape)
#print("y shape:", y.shape)

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
#summary(model, input_size=(X.shape[1],))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- 4. Train and save the model
def train(X, y, epochs, batch_size):
    
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
    return model


#train(X_train, y_train, epochs=200, batch_size=32)
#torch.save(model, "filter_mlp_model_full.pth")
gemt_model = torch.load("filter_mlp_model_full.pth", weights_only=False)
model = gemt_model


# ---- 5. Evaluation
model.eval()
with torch.no_grad():
    Y = model(dumm)
print(Y)
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()
    mse = np.mean((preds - y_test.numpy()) ** 2)
print(f"\nTest MSE: {mse:.6f}")



