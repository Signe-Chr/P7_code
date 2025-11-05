import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
import Dataset_generator_script as dgs
from tqdm import tqdm
from Dataset_class import CustomDataset, L, J
from multiprocessing import cpu_count

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

def compute_H_matrix(rir_array, fs=16000, n_fft=None):
    """
    Compute the frequency-domain transfer matrix H[k]
    from a set of impulse responses.

    Parameters
    ----------
    rir_array : np.ndarray, shape (n_mics, n_srcs, n_samples)
        Time-domain impulse responses for each mic–source pair.
        rir_array[m, s, :] = impulse response from source s to mic m.
    fs : int, optional
        Sampling frequency in Hz (default: 16000).
    n_fft : int, optional
        FFT length. If None, uses next power of 2 above rir length.

    Returns
    -------
    H : np.ndarray, shape (n_mics, n_srcs, n_freqs)
        Frequency response matrix for all microphone–source pairs.
    freqs : np.ndarray
        Frequency vector (in Hz) for the frequency bins.
    """
    # --- Input validation ---
    if rir_array.ndim != 3:
        raise ValueError(f"Expected rir_array of shape (n_mics, n_srcs, n_samples), got {rir_array.shape}")

    n_mics, n_srcs, n_samples = rir_array.shape

    # --- Choose FFT length ---
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))  # next power of 2

    n_freqs = n_fft // 2 + 1

    # --- Allocate frequency-domain matrix ---
    H = np.zeros((n_mics, n_srcs, n_freqs), dtype=np.complex128)

    # --- Compute FFT for each mic–source pair ---
    for m in range(n_mics):
        for s in range(n_srcs):
            h = rir_array[m, s, :]
            H[m, s, :] = np.fft.rfft(h, n=n_fft)

    # --- Frequency axis ---
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    return H, freqs

def C_i(AC_des, w_AC, AC_tilde):
    #print(np.real(AC_des * w_AC - AC_tilde))
    return torch.max(torch.tensor(0), torch.real(AC_des * w_AC - AC_tilde))
    
def w_ac(center_frequency, ref_frequency: float = 100.0, 
                beta: float = 1.0, min_weight: float = 1.0) -> list:
    """
    Calculates the frequency-dependent weight function (w_AC) for acoustic contrast.

    This function prioritizes low frequencies by assigning a higher weight 
    to lower frequency bands, ensuring the optimization loop focuses on the 
    most challenging bands first.

    The formula used is: w_AC = max(min_weight, (ref_frequency / f_i)^beta)

    Args:
        center_frequencies: List of center frequencies (in Hz) for the bands.
        ref_frequency: The reference frequency (Hz), typically the lowest 
                    frequency in the analysis range. This frequency will 
                    have the weight determined by 1/min_weight.
        beta: The exponent that controls the decay rate of the weight 
            (a higher beta means faster decay). Typical values are 0.5 to 1.5.
        min_weight: The minimum weight value allowed (usually 1.0 to ensure
                    the desired AC is at least met in high frequencies).

    Returns:
        A list of weights (w_AC) corresponding to the input frequencies.
    """
    
    # Convert inputs to NumPy arrays for vectorized calculation
    #f_i = torch.asarray(center_frequencies)
    
    # Calculate the ratio raised to the power beta
    weight_ratios = ((ref_frequency / center_frequency) ** beta)
    
    # Ensure the weight never drops below the specified minimum weight
    w_ac = max(weight_ratios, min_weight)
    #print(w_ac)
    return w_ac#.tolist()

def AC_tilde(H_B, H_D, g, M_B, M_D):
    g_col = g.unsqueeze(-1)
    H_B_d = H_B.to(g.dtype)#.detach()
    H_D_d = H_D.to(g.dtype)#.detach()
    
    # Calculate Energy in Bright Zone (E_B = ||H_B * g||^2) and Dark Zone (E_D = ||H_D * g||^2)
    E_B = torch.sum(torch.matmul(H_B_d, g_col).abs().pow(2))
    E_D = torch.sum(torch.matmul(H_D_d, g_col).abs().pow(2))

    return (M_D / M_B) * (E_B / E_D)

def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs and filters with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        filter_q: [n_srcs, filter_len]
        x_input: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = x_input.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
    # Zero pad x_input for convolution
    x_input_padded = F.pad(x_input, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            # Combined filter impulse response: h_combined = RIR * filter_q (via standard convolution)
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0) # [1, 1, n_rir_samples]
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0) # [1, 1, filter_len]
            
            # --- CRITICAL FIX: SWAP INPUT/KERNEL FOR CONV1D ---
            # Since n_rir_samples (512) < filter_len (1024), we must swap them for F.conv1d.
            # Convolution is commutative: rir * q = q * rir
            h_combined = F.conv1d(q_s, rir_m_s, padding=0).squeeze()
            
            # Convolve h_combined with input signal x (x_input) using FFT
            
            # Pad h_combined to ensure final output length matches 'output_len'
            h_combined_padded = F.pad(h_combined, (0, output_len - h_combined.shape[0]), 'constant', 0)
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(x_input_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    return p

def L_2_loss(test_filter_flat: torch.Tensor, candidate_filter_flat: torch.Tensor):
    """Cosine distance between two flattened filters."""
    y_test_norm = F.normalize(test_filter_flat, p=2, dim=1)
    y_cand_norm = F.normalize(candidate_filter_flat, p=2, dim=1)
    similarity = torch.mm(y_test_norm, y_cand_norm.T)
    cosine_distance = 1 - similarity.squeeze()
    return cosine_distance

def L_3_loss(test_filter_reshaped: torch.Tensor, candidate_filter_reshaped: torch.Tensor,
                                  rir_test: torch.Tensor, rir_train: torch.Tensor, 
                                  x_input: torch.Tensor, B_idx: list) -> torch.Tensor:
    """
    Compute MSPE (Mean Squared Pressure Error) only in the Bright Zone (B_idx)
    between the desired pressure (from test filter/RIR) and the predicted pressure 
    (from candidate filter/train RIR).
    """
    
    # 1. Calculate Desired Pressure (Reference: Test Filter + Test RIR)
    p_des_full = compute_pressure_with_input(rir_test, test_filter_reshaped, x_input) # [n_mics, n_samples]
    p_des_B = p_des_full[B_idx] # [M_B, n_samples]

    # 2. Calculate Predicted Pressure (Candidate: Candidate Filter + Train RIR)
    p_pred_full = compute_pressure_with_input(rir_train, candidate_filter_reshaped, x_input) # [n_mics, n_samples]
    p_pred_B = p_pred_full[B_idx] # [M_B, n_samples]

    # 3. Compute MSE
    msep_loss = torch.mean((p_pred_B - p_des_B) ** 2)
    return msep_loss

def L_4_loss(q_opt, rir, x_input, fcentres, H, bright_indices, dark_indices, M_B, M_D):
    fd = torch.tensor(2**(1/6))
    delta_f = dgs.fs_target/dgs.J
    L_4 = 0
    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd
        g = torch.fft.fft(q_opt, axis = 0)
        p_des_full = compute_pressure_with_input(rir, q_opt, x_input)
        p_des_B = p_des_full[bright_indices]
        p_des_D = p_des_full[dark_indices]
        
        # AC_des is generally calculated in terms of pressure magnitude difference or ratio (in linear scale)
        E_des_B = torch.sum(p_des_B ** 2)
        E_des_D = torch.sum(p_des_D ** 2)
        AC_des = (M_D / M_B) * (E_des_B / E_des_D) if E_des_D.item() != 0 else torch.tensor(1e10)

        k_low = int(torch.ceil(f_low/delta_f))
        k_high = int(torch.ceil(f_high/delta_f))
        L_4_ = 0
        for k in range(k_low, k_high):
            AC_sim = AC_tilde(H[bright_indices][:,:,k], H[dark_indices][:,:,k], g[:,k], M_B, M_D)
            w_AC = w_ac(freq, ref_frequency=100, beta=1, min_weight=1)
            C = C_i(AC_des, w_AC, AC_sim)
            L_4_ += C**2
        L_4 += torch.sqrt(L_4_)
        del L_4_
    return L_4

fcentres = torch.tensor([1000, 2000])

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
            loss = L_1_loss(outputs, batch_y) + L_2_loss(pre_flat_outputs, pre_flat_batch_y) + L_3_loss(outputs, batch_y, batch_IR, batch_IR, wav, bright_batch) + L_4_loss(batch_y, batch_IR, wav, fcentres, H, bright_batch, dark_batch, len(batch[2]), len(batch[3]))
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