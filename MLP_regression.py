import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
import Dataset_generator_script as dgs
from tqdm import trange

L = 3       # Loudspeaker
J = 1024    # Filter order

# ---- 1. Load data
def load_data(dataset = "VAST_filter_archive.npy"):
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

def toeplitz_matrix(h: np.ndarray, block_len: int) -> np.ndarray:
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
            
    return torch.tensor(H_multi)

def L_1_loss(q_opt, fcentres, M_B, H):
    g = torch.fft.fft(q_opt, axis = 0)
    target_pressure = torch.abs(g)*1.3 # revurderes
    fd = 2**(1/6)
    delta_f = dgs.fs_target/dgs.J
    L_1 = 0

    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd

        k_low = int(np.ceil(f_low/delta_f))
        k_high = int(np.ceil(f_high/delta_f))

        L_1_ = 0

        for m in range(M_B):
            temp = 0
            for k in range(k_low, k_high):
                H_B_slice_complex = torch.tensor(H[:,:,k]).to(g.dtype)
                H_B_tilde = torch.matmul(H_B_slice_complex, g)
                temp += (torch.linalg.norm(H_B_tilde[m,:], ord=1)-torch.linalg.norm(target_pressure[:,k], ord=1))**2
            L_1_ += torch.sqrt(temp)
            del temp
        L_1 += L_1_
        del L_1_
    return L_1

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

def L_2_loss(q_opt, fcentres, H_B, H_D, M_B, M_D):
    fd = torch.tensor(2**(1/6))
    delta_f = dgs.fs_target/dgs.J
    L_2 = 0
    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd
        g = torch.fft.fft(q_opt, axis = 0)
        AC_des = 10**(-50/10)#5.079192938063992e-07

        k_low = int(torch.ceil(f_low/delta_f))
        k_high = int(torch.ceil(f_high/delta_f))
        L_2_ = 0
        for i in range(k_low, k_high):
            AC_sim = AC_tilde(H_B[:,:,i], H_D[:,:,i], g[:,i], M_B, M_D)
            w_AC = w_ac(freq, ref_frequency=100, beta=1, min_weight=1)
            C = C_i(AC_des, w_AC, AC_sim)
            L_2_ += C**2
        L_2 += torch.sqrt(L_2_)
        del L_2_
    return L_2

def energy_tilde(q_opt, H_time, N_time_steps, mic_index, speaker_index):

    e_b = torch.tensor(0.0, dtype=H_time.dtype)
    
    if isinstance(H_time, np.ndarray):
        H_time = torch.from_numpy(H_time)
    
    

    for n in range(N_time_steps-500):
        H_n_vector = H_time[n, mic_index, speaker_index, :].to(q_opt.dtype).detach()

        y_n = torch.dot(q_opt, H_n_vector)
        
        e_b += torch.square(y_n)
        
    #print(f"Calculated Energy e_b: {e_b.item()}")
    return e_b

def energy(H_time, mic_index: int, speaker_index: int) -> torch.Tensor:

    H_ms = H_time[:, mic_index, speaker_index, :]

    e_b = torch.sum(H_ms**2)
    
    return e_b

def L_3_loss(q_opt, H_time, N_time_steps = dgs.N):
    L_3 = torch.tensor(0.0)
    
    for m in bright_zone_mics_index:
        mm = torch.tensor(0.0) 
        
        for s in range(n_srcs):
            E_tilde_num = energy_tilde(q_opt[s], H_time, N_time_steps, m, s)
            E_tilde_den = energy_tilde(q_opt[s], H_time, N_time_steps, m, -1) 
            
            E_num = energy(H_time, m, s)
            E_den = energy(H_time, m, -1)
            
            if E_tilde_den.item() == 0 or E_den.item() == 0:
                diff_squared = torch.tensor(0.0)
            else:
                diff_squared = (E_tilde_num / E_tilde_den - E_num / E_den)**2

            mm += diff_squared

        L_3 += torch.sqrt(mm)

    return L_3

fcentres = torch.tensor([1000, 2000])

# ---- 4. Train and save the model
def train(X, y, IRs, bright_mics, dark_mics, epochs, model, dev, batch_size=1):
     
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    c = nn.MSELoss()

    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        total_loss = 0.0

        loop = trange(0, X.size(0), batch_size, desc=f"Epoch {epoch+1}/{epochs}")
        for i in loop:
            idx = permutation[i:i + batch_size]
            batch_X, batch_y = X[idx].to(dev), y[idx].to(dev)
            batch_y = batch_y.reshape(3, 1024)

            optimizer.zero_grad()
            outputs = model(batch_X)
            outputs = outputs.reshape(3, 1024)
            H, _ = compute_H_matrix(IRs[i])
            H_B = torch.from_numpy(H[bright_mics])
            H_D = torch.from_numpy(H[dark_mics])

            H_time = compute_multi_toeplitz(IRs[i], len(batch_y[0]))
            loss = c(outputs, batch_y) + L_1_loss(batch_y, fcentres, len(bright_mics), H) + L_2_loss(batch_y, fcentres, H_B, H_D, len(bright_mics), len(dark_mics)) + L_3_loss(batch_y, H_time, N_time_steps = dgs.N)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(f"Loss: {total_loss:.4f}")
        #if (epoch + 1) % 20 == 0:
        #print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return model

def review_data():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = os.listdir("VAST_filter_archive")
    for key in keys:
        X_train, X_test, y_train, y_test, bright_zone_mics_index, dark_zone_mics_index, n_srcs, IR_list = load_data()
    model = FilterNet(input_size=X_train.shape[1], output_size=y_train.shape[1]).to(device)
    model = train(X_train, y_train, IR_list, bright_zone_mics_index, dark_zone_mics_index, epochs=20, dev=device, batch_size=32, model=model)
    #torch.save(model, "filter_mlp_model_full.pth")
    torch.save(model.state_dict(), "mlp_weights_r.pth")

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