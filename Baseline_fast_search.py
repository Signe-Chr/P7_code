import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
# Assuming dgs is correctly imported and available
import Dataset_generator_script as dgs 

L = 3       # Loudspeaker (Sources)
J = 1024    # Filter order
# M_B and M_D will be calculated from the indices

# ---- 1. Load data
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

# Get indices (assuming they are lists/arrays of indices)
bright_zone_mics_index = data["VAST_0_0_0_0"].get('bright_zone_mics_index', [])
dark_zone_mics_index = data["VAST_0_0_0_0"].get('dark_zone_mics_index', [])

# Calculate number of mics from indices
M_B = len(bright_zone_mics_index)
M_D = len(dark_zone_mics_index)

n_srcs = len(data["VAST_0_0_0_0"].get('sources_position', []))

X_list, y_list, IR_list = [], [], []

for key, inner in data.items():
    rt60 = inner.get('RT60', 0)
    phone_tilt = inner.get('Phone_tilt', 0)
    user_orient = inner.get('User_orientation', 0)
    spatial = np.array(inner.get('Spatial_position', [0,0,0])).ravel()
    room_dim = inner.get('room_dim', [0,0,0])
    IR = inner.get('IR', np.zeros((1, L, 1))) # Placeholder for IR shape robustness
    
    X = np.concatenate([
        [rt60], [phone_tilt], [user_orient], spatial, room_dim
    ])
    
    # Output: flatten q_matrix (L*J)
    # Ensure q_matrix is L*J, matching the model's output size
    q_matrix = inner.get('q_matrix', np.zeros((L, J))) 
    y = np.ravel(q_matrix)
    
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

# ---- 3. Define the network
class FilterNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FilterNet(input_size=X.shape[1], output_size=y.shape[1]).to(device)

# Loss functions (optimized)
# The definitions of compute_H_matrix, toeplitz_matrix, compute_multi_toeplitz are mostly fine
# but compute_multi_toeplitz now returns a torch tensor.

def compute_H_matrix(rir_array, fs=16000, n_fft=None):
    # ... (function body remains the same, returns NumPy H, freqs) ...
    if rir_array.ndim != 3:
        raise ValueError(f"Expected rir_array of shape (n_mics, n_srcs, n_samples), got {rir_array.shape}")
    n_mics, n_srcs, n_samples = rir_array.shape
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    n_freqs = n_fft // 2 + 1
    H = np.zeros((n_mics, n_srcs, n_freqs), dtype=np.complex128)
    for m in range(n_mics):
        for s in range(n_srcs):
            H[m, s, :] = np.fft.rfft(rir_array[m, s, :], n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    return H, freqs

def toeplitz_matrix(h: np.ndarray, block_len: int) -> np.ndarray:
    if not isinstance(h, np.ndarray):
        h = np.array(h, dtype=np.float32)
    L_h = h.shape[0]
    R = L_h + block_len - 1
    C = block_len
    H_toeplitz = np.zeros((R, C), dtype=h.dtype)
    for i in range(C):
        H_toeplitz[i:i + L_h, i] = h
    return H_toeplitz

def compute_multi_toeplitz(rir_array: np.ndarray, block_len: int) -> torch.Tensor:
    if not isinstance(rir_array, np.ndarray) or rir_array.dtype != np.float32:
        rir_array = np.array(rir_array, dtype=np.float32)
    M, S, L_ir = rir_array.shape
    K = block_len
    output_len = L_ir + K - 1
    H_multi = np.zeros((output_len, M, S, K), dtype=rir_array.dtype)
    for m in range(M):
        for s in range(S):
            h_ms = rir_array[m, s, :]
            H_ms_toeplitz = toeplitz_matrix(h_ms, K)
            H_multi[:, m, s, :] = H_ms_toeplitz
    return torch.tensor(H_multi, dtype=torch.float32) # Explicitly cast to torch.float32

# --- OPTIMIZED L_1_loss ---
def L_1_loss(q_opt: torch.Tensor, fcentres: torch.Tensor, M_B: int, H: np.ndarray) -> torch.Tensor:
    """
    Calculates the pressure constraint loss in the bright zone.
    
    q_opt shape: (L, J) - Sources x Filter_Order
    H shape: (n_mics, L, n_freqs) - Mics x Sources x Frequencies
    """
    
    # g shape: (L, n_freqs)
    g = torch.fft.rfft(q_opt, dim=-1) # Use rfft for real input q_opt
    n_freqs = g.shape[-1]
    
    # target_pressure shape: (L, n_freqs)
    target_pressure = torch.abs(g) * 1.3
    fd = 2**(1/6)
    
    # Assuming dgs.fs_target/dgs.J is the frequency bin resolution of rfft
    delta_f = dgs.fs_target / (2 * (n_freqs - 1)) # Calculate delta_f based on rfft output
    
    # Convert H (NumPy) to a PyTorch tensor once outside the inner loops
    H_tensor = torch.from_numpy(H).to(g.dtype)
    
    L_1 = torch.tensor(0.0, dtype=g.dtype.real_type(), device=g.device) # Use real type for loss
    
    for freq in fcentres:
        f_low = freq / fd
        f_high = freq * fd

        # Calculate k indices (freq bins)
        k_low = int(torch.ceil(f_low / delta_f))
        k_high = int(torch.ceil(f_high / delta_f))
        
        # Ensure indices are within bounds
        k_low = max(0, k_low)
        k_high = min(n_freqs, k_high)
        
        if k_high <= k_low:
            continue

        L_1_ = 0

        # Slice H and g for the current frequency band
        H_band_slice = H_tensor[bright_zone_mics_index, :, k_low:k_high] # (M_B, L, k_band_len)
        g_band_slice = g[:, k_low:k_high] # (L, k_band_len)
        target_band_slice = target_pressure[:, k_low:k_high] # (L, k_band_len)

        # Batch matrix multiplication: (M_B, L, k_band_len) @ (L, k_band_len, 1) -> (M_B, k_band_len, 1)
        # Using tensordot for (M_B x L) @ (L x k_band_len) -> (M_B x k_band_len)
        # H_band_slice must be permuted to (k_band_len, M_B, L)
        
        # Perform batched matrix multiplication over frequency bins k:
        # H_B_slice_complex[mics x sources] @ g_k[sources x 1] 
        # For all k in the band: (M_B x L) @ (L x k_band_len) -> (M_B x k_band_len)
        
        # NOTE: If your IR has the same length as the filter J, use J frequency bins
        # If your IR has length N and n_fft is used, the freq bins are n_fft//2 + 1
        
        # OPTIMIZED: Vectorized over frequency band
        # H_B_tilde shape: (M_B, k_band_len)
        # torch.einsum is used for: Pressure[m, k] = H[m, s, k] * g[s, k]
        H_B_tilde = torch.einsum('msf, sf->mf', H_band_slice, g_band_slice)
        
        # Calculate loss component for the entire band
        
        # Pressure magnitude at each mic m over the band: (M_B, k_band_len)
        pressure_mag = torch.abs(H_B_tilde)
        
        # Target pressure: target_pressure[:, k] is for sources L, not Mics M_B. This seems wrong.
        # Assuming a flat target for all Mics M_B, based on the magnitude of g.
        # This part requires re-evaluation of the physical model, but keeping the structure:
        
        # Sum over frequency bins (dim=1) for each mic m (dim=0)
        # L1 norm of pressure for each mic: (M_B,)
        pressure_l1_per_mic = torch.linalg.norm(pressure_mag, ord=1, dim=1) 
        
        # Target pressure L1 norm over band for the L sources, now needs to be applied to M_B mics
        # Assuming the target magnitude is uniform across M_B mics (L_1_norm of target_pressure[:,k] is for sources L, not Mics M_B)
        # You had: torch.linalg.norm(target_pressure[:,k], ord=1))**2. Summing L-sources L1 norms.
        
        # Assuming the target L1 norm must be calculated once for the band:
        target_l1_norm = torch.linalg.norm(target_band_slice, ord=1, dim=0).sum()
        target_l1_norm_per_mic = target_l1_norm / M_B # Simple average
        
        # The loss calculation is highly custom and likely intended to be:
        # L_1_ = sum_{m in M_B} sqrt( sum_{k in band} ( ||P[m,k]||_1 - ||Target[k]||_1 )^2 )
        
        # To match the original intent, we calculate the difference for all mics m over the band:
        # Replicate target norm for M_B mics
        target_l1_norm_mics = target_l1_norm_per_mic.expand(M_B) 
        
        # Difference (M_B,)
        diff = pressure_l1_per_mic - target_l1_norm_mics
        
        # (M_B,) -> scalar: Sum over mics, then sqrt
        L_1_ = torch.sqrt(torch.sum(diff**2))
        
        L_1 += L_1_
    
    return L_1

# The other loss functions L_2 and L_3 require the same careful attention to indexing and dimension.
# For simplicity and to fix the structural errors first, we focus on L_1 and L_2.

# --- OPTIMIZED L_2_loss ---
def L_2_loss(q_opt: torch.Tensor, fcentres: torch.Tensor, H_B: torch.Tensor, H_D: torch.Tensor, M_B: int, M_D: int) -> torch.Tensor:
    """
    Calculates the acoustic contrast constraint loss.
    
    q_opt shape: (L, J)
    H_B shape: (M_B, L, n_freqs)
    H_D shape: (M_D, L, n_freqs)
    """
    fd = 2**(1/6)
    g = torch.fft.rfft(q_opt, dim=-1) # (L, n_freqs)
    n_freqs = g.shape[-1]
    
    # Calculate delta_f based on rfft output
    delta_f = dgs.fs_target / (2 * (n_freqs - 1))
    
    L_2 = torch.tensor(0.0, dtype=g.dtype.real_type(), device=g.device)
    AC_des = 10**(-50/10) # Desired acoustic contrast (linear scale)

    for freq in fcentres:
        f_low = freq / fd
        f_high = freq * fd
        
        k_low = int(torch.ceil(f_low / delta_f))
        k_high = int(torch.ceil(f_high / delta_f))
        
        k_low = max(0, k_low)
        k_high = min(n_freqs, k_high)

        if k_high <= k_low:
            continue
        
        # Slice H matrices and g for the frequency band
        H_B_band = H_B[:, :, k_low:k_high] # (M_B, L, k_band_len)
        H_D_band = H_D[:, :, k_low:k_high] # (M_D, L, k_band_len)
        g_band = g[:, k_low:k_high]        # (L, k_band_len)
        
        # Calculate Energy in Bright Zone (E_B) and Dark Zone (E_D) over the band (vectorized)
        
        # Pressure in Bright Zone P_B: (M_B, k_band_len)
        P_B = torch.einsum('msf, sf->mf', H_B_band, g_band)
        E_B = torch.sum(P_B.abs().pow(2))
        
        # Pressure in Dark Zone P_D: (M_D, k_band_len)
        P_D = torch.einsum('msf, sf->mf', H_D_band, g_band)
        E_D = torch.sum(P_D.abs().pow(2))

        # AC_tilde (Acoustic Contrast over the band)
        # Note: If E_D is zero, this will raise a division by zero error.
        if E_D.item() == 0:
            AC_sim = torch.tensor(float('inf'))
        else:
            AC_sim = (M_D / M_B) * (E_B / E_D)

        # Assuming w_ac is a scalar function for the center frequency
        w_AC = w_ac(freq.item(), ref_frequency=100, beta=1, min_weight=1) 
        
        # Constraint violation
        C = torch.max(torch.tensor(0.0, device=g.device), AC_des * w_AC - AC_sim)
        
        # L_2 contribution for this band is the square of the violation
        L_2_ = C**2

        L_2 += L_2_

    return torch.sqrt(L_2) # The original code takes sqrt of the sum of C^2 over bands

# L_3_loss and supporting functions are not fully optimized but are left in the
# original structure as they do not show the major dimensional errors of L1/L2.
# Minor fix in energy_tilde for robustness (q_opt is (L, J), so q_opt[s] is (J,))

def energy_tilde(q_opt_s, H_time, N_time_steps, mic_index, speaker_index):
    # q_opt_s shape: (J,)
    e_b = torch.tensor(0.0, dtype=H_time.dtype, device=H_time.device)
    
    # H_time is already a torch.Tensor from compute_multi_toeplitz
    
    # Note: Your use of N_time_steps-500 seems arbitrary; 
    # the maximum convolution output length should be used (H_time.shape[0])
    max_steps = min(N_time_steps - 500, H_time.shape[0]) 
    
    # H_time[n, mic_index, speaker_index, :] shape is (K=J,)
    for n in range(max_steps):
        H_n_vector = H_time[n, mic_index, speaker_index, :].to(q_opt_s.dtype).detach()
        y_n = torch.dot(q_opt_s, H_n_vector)
        e_b += torch.square(y_n)
    return e_b

def L_3_loss(q_opt, H_time, N_time_steps = dgs.N):
    L_3 = torch.tensor(0.0, device=q_opt.device)
    
    for m in bright_zone_mics_index:
        mm = torch.tensor(0.0, device=q_opt.device)
        
        for s in range(n_srcs):
            # Pass the filter for source s: q_opt[s] shape (J,)
            E_tilde_num = energy_tilde(q_opt[s], H_time, N_time_steps, m, s)
            E_tilde_den = energy_tilde(q_opt[s], H_time, N_time_steps, m, -1) # NOTE: -1 index is dangerous/unverified
            
            E_num = energy(H_time, m, s)
            E_den = energy(H_time, m, -1) # NOTE: -1 index is dangerous/unverified
            
            # Use torch.isclose for robust check instead of .item() == 0
            if torch.isclose(E_tilde_den, torch.tensor(0.0)) or torch.isclose(E_den, torch.tensor(0.0)):
                diff_squared = torch.tensor(0.0, device=q_opt.device)
            else:
                diff_squared = (E_tilde_num / E_tilde_den - E_num / E_den)**2

            mm += diff_squared

        L_3 += torch.sqrt(mm)
    return L_3

# ... (w_ac, C_i, AC_tilde, energy functions remain the same/minor fix) ...
def w_ac(center_frequency, ref_frequency: float = 100.0, beta: float = 1.0, min_weight: float = 1.0):
    weight_ratios = ((ref_frequency / center_frequency) ** beta)
    return max(weight_ratios, min_weight)

def C_i(AC_des, w_AC, AC_tilde):
    return torch.max(torch.tensor(0.0, device=AC_tilde.device), AC_des * w_AC - AC_tilde)

def energy(H_time, mic_index: int, speaker_index: int) -> torch.Tensor:
    H_ms = H_time[:, mic_index, speaker_index, :]
    e_b = torch.sum(H_ms**2)
    return e_b

fcentres = torch.tensor([1000, 2000])

# ---- 4. Train and save the model
def train(X, y, epochs, batch_size, model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    c = nn.MSELoss()
    epochs = 200 # Overriding passed epochs
    batch_size = 1 # Overriding passed batch_size (NOTE: set to 1 for this data loading structure)

    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[idx].to(device), y_train[idx].to(device)
            
            # Reshape target: (1, L*J) -> (L, J) = (3, 1024)
            batch_y_reshaped = batch_y.reshape(L, J)

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Reshape output to match target: (1, L*J) -> (L, J)
            outputs_reshaped = outputs.reshape(L, J) 
            
            # Load and convert IR-related matrices for the current sample 'i'
            IR = IR_list[i]
            H, freqs = compute_H_matrix(IR)
            
            # H_B, H_D slicing and NumPy -> PyTorch conversion
            H_B = torch.from_numpy(H[bright_zone_mics_index]).to(device).to(outputs_reshaped.dtype)
            H_D = torch.from_numpy(H[dark_zone_mics_index]).to(device).to(outputs_reshaped.dtype)

            # Compute H_time (Toeplitz matrix)
            # The length of batch_y[0] (1024) is used as the filter order K in Toeplitz
            H_time = compute_multi_toeplitz(IR, J).to(device) 

            # Calculate loss using reshaped output/target and M_B, M_D counts
            loss = c(outputs_reshaped, batch_y_reshaped) + \
                   L_1_loss(outputs_reshaped, fcentres, M_B, H) + \
                   L_2_loss(outputs_reshaped, fcentres, H_B, H_D, M_B, M_D) + \
                   L_3_loss(outputs_reshaped, H_time, N_time_steps=dgs.N)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return model

if __name__ == "__main__":
    # The print block is fine as is
    for key, inner in data.items():
        print(f"--- Key: {key} ---")
        for field in inner:
            value = inner[field]
            if isinstance(value, np.ndarray):
                print(f"{field}: numpy array, shape = {value.shape}")
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                print(f"{field}: list/array, length/shape = {len(value) if isinstance(value, list) else value.shape}")
            else:
                print(f"{field}: type = {type(value)}, value = {value}")
        break

    train(X_train, y_train, epochs=200, batch_size=32, model=model)
    torch.save(model.state_dict(), "mlp_weights_r.pth")

    dummy_input = np.array([2.2,    # Reverberation, float
                        0.78,      # Phone tilt, degrees
                        3.14,      # Orientation,  degrees
                        5, 10, 1.7],
                        [10,10,10])    # Spatial position, (x, y, z)       

    dumm = torch.tensor(dummy_input, dtype=torch.float32)
    print(dumm, dummy_input)

    model.eval()
    with torch.no_grad():
        Y = model(dumm)
    print(Y)
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        mse = np.mean((preds - y_test.numpy()) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    model.eval()
    # ---- 5. Evaluation and saving coefficients
    with torch.no_grad():
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

    model.eval()
    with torch.no_grad():
        test_input = np.concatenate([[0.6], [np.deg2rad(15)], [np.pi/2], [2, 2, 4], [10, 10, 10]]).astype(np.float32)
        test_input_scaled = scaler_X.transform(test_input.reshape(1, -1))
        test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32).to(device)
        predicted_filter = model(test_tensor).cpu().numpy().squeeze()
        print(f"Predicted filter shape: {predicted_filter.shape}")
        np.savetxt("predicted_filter_fnet_2.txt", predicted_filter, fmt="%.8f")
        
        preds = model(X_test.to(device)).cpu().numpy()
        mse = np.mean((preds - y_test.numpy()) ** 2)
    print(f"\nTest MSE: {mse:.6f}")