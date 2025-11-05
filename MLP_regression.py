import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
import Dataset_generator_script as dgs
from tqdm import tqdm
from Dataset_class import CustomDataset, L, J
from multiprocessing import cpu_count
from Loss_functions import L_2_loss, L_3_loss, L_4_loss, compute_H_matrix

# ---- 1. Load data
def load_data(dataset = "VAST_filter_archive.npy"):
    # THIS FUNCTION IS NO LONGER USED IN THIS SCRIPT
    data = np.load(dataset, allow_pickle=True).item()
    bright_zone_mics_index = data["VAST_0_0_0_0"]['bright_zone_mics_index']
    dark_zone_mics_index = data["VAST_0_0_0_0"]['dark_zone_mics_index']
    n_srcs = len(data["VAST_0_0_0_0"]['sources_position'])
    
    X_list, y_list = [], []
    IR_list = []
    for key, inner in data.items():
        # Robust håndtering af input features (fallback til 0 hvis mangler)
        rt60 = inner.get('RT60', 0)                         # 2.5
        phone_tilt = inner.get('Phone_tilt', 0)             # I radianer: 0.261, 0.785, 1.309
        user_orient = inner.get('User_orientation', 0)      # I radianer: 0, 1.57, 3.14, 4.71
        spatial = inner.get('Spatial_position', [0,0,0])    # (x, y, z): (5, 5 ,1.7) betyder i midten af rummet og i højde 1.7m
        spatial = np.array(spatial).ravel()                 # flad ud til 1D
        room_dim = inner.get('room_dim', [0,0,0])
        IR = inner.get('IR', [0,0,0])
        
        X = np.concatenate([
            [rt60],
            [phone_tilt],
            [user_orient],
            spatial,
            room_dim
        ])
        
        # Output: flatten q_matrix
        y = np.ravel(inner.get('q_matrix', np.zeros(L*J)))
        IR_list.append(IR)
        X_list.append(X)
        y_list.append(y)

    X = np.stack(X_list)
    y = np.stack(y_list)

    # ---- 2. Train/test split and scaling
    scaler_X = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, X_test, y_train, y_test, bright_zone_mics_index, dark_zone_mics_index, n_srcs, IR_list

# ---- 3. Define the network

'''def toeplitz_matrix(h: np.ndarray, block_len: int) -> np.ndarray:
    """Helper to construct a single Toeplitz matrix."""
    # Ensure h is a NumPy array with a standard float dtype
    if not isinstance(h, np.ndarray):
        h = np.array(h, dtype=np.float32)
        
    L = h.shape[0]
    R = L + block_len - 1
    C = block_len
    
    # Use h.dtype, which is now guaranteed to be a NumPy dtype
    H_toeplitz = np.zeros((R, C), dtype=h.dtype)
    
    for i in range(C):
        H_toeplitz[i:i + L, i] = h
        
    return H_toeplitz

def compute_multi_toeplitz(rir_array: np.ndarray, block_len: int) -> np.ndarray:
    """
    Computes a 4D tensor containing the Toeplitz matrix for every
    Mic-Source pair.

    Shape: (Output_Time, Mic, Source, Input_Time/Delay_Index)
    """
    # CRITICAL FIX: Ensure input is a standard NumPy array with a native dtype.
    # The warning "Casting complex values to real discards the imaginary part" 
    # suggests your 'rir' variable might contain complex data or be a mix of types.
    # We explicitly convert it to real, single-precision floats (np.float32).
    if not isinstance(rir_array, np.ndarray) or rir_array.dtype != np.float32:
        rir_array = np.array(rir_array, dtype=np.float32)

    M, S, L = rir_array.shape
    K = block_len
    
    output_len = L + K - 1
    
    # Now, rir_array.dtype is guaranteed to be np.float32, resolving the TypeError.
    H_multi = np.zeros((output_len, M, S, K), dtype=rir_array.dtype)
    
    for m in range(M):
        for s in range(S):
            h_ms = rir_array[m, s, :]
            # We pass the guaranteed clean NumPy array slice to the helper
            H_ms_toeplitz = toeplitz_matrix(h_ms, K)
            
            H_multi[:, m, s, :] = H_ms_toeplitz
            
    return torch.tensor(H_multi)'''

# ---- 4. Train and save the model
def train(data, wav, epochs, model, dev):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    L_1_loss = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0

        loop = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch_X, pre_flat_batch_y = batch[0].to(dev, dtype=torch.float32), batch[1].to(dev, dtype=torch.float32)
            batch_y = pre_flat_batch_y.reshape(L, J)
            batch_IR = batch[5][0]
            bright_batch = batch[2][0]
            dark_batch = batch[3][0]

            optimizer.zero_grad()
            pre_flat_outputs = model(batch_X)
            outputs = pre_flat_outputs.reshape(L, J)
            H = torch.from_numpy(compute_H_matrix(batch_IR)[0]).to(dev)
            #H_B = H[bright_batch][0]
            #H_D = H[batch[3][0]][0]

            #H_time = compute_multi_toeplitz(batch_IR, len(batch_y[0])).to(dev)
            loss = L_1_loss(outputs, batch_y) + L_2_loss(pre_flat_outputs, pre_flat_batch_y) + L_3_loss(outputs, batch_y, batch_IR, batch_IR, wav, bright_batch) + L_4_loss(batch_y, batch_IR, wav, H, bright_batch, dark_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}")
        #if (epoch + 1) % 20 == 0:
        #print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return model

def review_data():
    #OUTDATED FUNCTION
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


if __name__ == "__main__":
    import torch
    import Cross_validation_models as cvm
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "VAST_filter_archive_730"
    from Dataset_generator_script import room_indices as ri
    full_data = os.listdir(data_dir)
    #data_points = []
    train_points = []
    #test_points = []
    for data in full_data:
        i = int(data.split("_")[1])
        if (i in ri) and (i not in ri[::4]):
            train_points.append(data)
            #data_points.append(data)
    #data_points = full_data[:1000]
    #train_files, test_files = train_test_split(data_points, test_size=0.2, random_state=42)
    trainset = CustomDataset(data_dir, train_points)
    #testset = CustomDataset(data_dir, test_points)
    p_features = len(trainset[0][0])
    out_features = len(trainset[0][1])
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=cpu_count()//3*2)
    #test_loader = DataLoader(testset, shuffle=False)
    wav = dgs.wav / np.max(np.abs(dgs.wav))
    model = train(train_loader, torch.from_numpy(wav).to(device), epochs=5, dev=device, model=cvm.model_.to(device))
    torch.save(model.state_dict(), f"MLP_regression.pth")
    """for i, model in enumerate(cvm.L[]):
        model = train(train_loader, epochs=5, dev=device, model=model.to(device))
        torch.save(model.state_dict(), f"Regression_cross_validation_models/MLP_regression_cross_{i}.pth")"""

    #torch.save(model, "filter_mlp_model_full.pth")

    

    # ---- 5. Evaluation and saving coefficients
    """with torch.no_grad():
        # ---- Custom input as requested
        test_input = np.concatenate([[0.6], [np.deg2rad(15)], [np.pi/2], [2, 2, 4]]).astype(np.float32)
        test_input_scaled = scaler_X.transform(test_input.reshape(1, -1))
        test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32).to(device)

        predicted_filter = model(test_tensor).cpu().numpy().squeeze()

        print(f"Predicted filter shape: {predicted_filter.shape}")
        print("First 10 coefficients:", predicted_filter[:10])

        # ---- Save coefficients
        np.savetxt("predicted_filter_fnet_2.txt", predicted_filter, fmt="%.8f")
        print(f"All {predicted_filter.size} coefficients saved to 'predicted_filter_fnet.txt'")

    # ---- Optional: compute test set MSE
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        mse = np.mean((preds - y_test.numpy()) ** 2)
    print(f"\nTest MSE: {mse:.6f}")

"""