import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy.signal import lfilter
import os
import VAST_dictionary_generator as vdg
import VISUALIZE_q_matrix as vq
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Signe_sang.wav")
fs, lyd_data = wavfile.read(file_path)

lyd_data=list(np.array(lyd_data[0:fs*1])/max(lyd_data))


q = torch.tensor(vq.q[0])
fcentre = torch.tensor([1000, 2000])
J = vdg.J


class ILZ_CNN_RIR(nn.Module):
    """
    CNN architecture for Source Zone Control (SZC) filter optimization.

    Parameters:
    M (int): Number of Microphones.
    S (int): Number of Sources (Loudspeakers).
    T (int): Number of filter coefficients per Loudspeaker (Filter length J).
    K (int): Length of the RIR input (Time dimension L).
    """
    def __init__(self, M, S, T, K):
        super(ILZ_CNN_RIR, self).__init__()
        self.M, self.S, self.T = M, S, T
        
        # NOTE: Input Shape: (Batch, 1, M, S, K)
        
        # Layer 1: Aggregates across Mics (M)
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=48, 
            kernel_size=(M, 1, 1), # Consumes M dimension
            padding=(0, 0, 0)
        )
        # Output shape after conv1: (Batch, 48, 1, S, K)
        
        # Layer 2: Aggregates across Sources (S)
        self.conv2 = nn.Conv3d(
            in_channels=48, out_channels=24, 
            kernel_size=(1, S, 1), # Consumes S dimension
            padding=(0, 0, 0)
        )
        # Output shape after conv2: (Batch, 24, 1, 1, K)
        
        # Calculate FC input size based on the remaining K dimension
        self.fc_input_size = 24 * K # This K must match max_length used in preparation

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 10)
        self.fc2 = nn.Linear(10, S * T) # Final output size: S * T

        # Activation Function
        self.activation = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        
    def forward(self, x):
        # Convolutional Block
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        
        # Flatten for FC layers: (batch, 24 * K)
        x = x.view(x.size(0), -1) 
        
        # Fully Connected Block
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape output to (batch, S, T) -> Filter coefficients q
        q = x.view(x.size(0), self.S, self.T)
        
        return q

model = ILZ_CNN_RIR(M=vdg.n_mics, S=vdg.n_srcs, T=J, K=J)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)  # Lower learning rate
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) #lr: learning rate, weight_decay: L2 regularization
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)


NN_INPUT, setup_information = vdg.NN_input(5)
sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index = setup_information[-1][0], setup_information[-1][1], setup_information[-1][2], setup_information[-1][3]
#[rir_tensor, rir_list]
# [sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index]

n_mics = len(mic_positions_list)
n_srcs = len(sources_position_list)


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
            
    return H_multi






def L_1_loss(q_opt, fcentres, M_B, H):
    g = torch.fft.fft(q_opt, axis = 0)
    target_pressure = torch.abs(g)*1.3 # revurderes
    fd = 2**(1/6)
    delta_f = vdg.fs_target/vdg.J
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
                H_B_slice_complex = H[:,:,k].to(g.dtype)
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
    delta_f = vdg.fs_target/vdg.J
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
    
    

    for n in range(N_time_steps):
        H_n_vector = H_time[n, mic_index, speaker_index, :].to(q_opt.dtype).detach()
        y_n = torch.dot(q_opt, H_n_vector)
        
        e_b += torch.square(y_n)
        
    #print(f"Calculated Energy e_b: {e_b.item()}")
    return e_b

def energy(H_time, mic_index: int, speaker_index: int) -> torch.Tensor:

    H_ms = H_time[:, mic_index, speaker_index, :]

    e_b = torch.sum(H_ms**2)
    
    return e_b

def L_3_loss(q_opt, H_time, N_time_steps = vdg.N):
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


def C_sk(G,k):
    hann_window = np.hanning(vdg.N)
    return max(abs(G[k])-hann_window[k], 0)

def L_4_loss(q_opt):
    L_4 = 0
    delta_f = vdg.fs_target/vdg.J
    k_1 = int(np.ceil(20/delta_f))
    k_2 = int(np.floor(20000/delta_f))
    k_s = int(np.ceil(vdg.fs_target/2/delta_f))
    G = np.fft.fft(q_opt, axis = 0)

    for s in range(n_srcs):
        G_s = G[s]
        c_sum = 0
        for k in range(k_1+1):
            c_sum += C_sk(G_s,k)**2
        for k in range(k_2, k_s+1):
            c_sum += C_sk(G_s,k)**2  
        L_4 += np.sqrt(c_sum)

    return L_4

def w_g(tau, tau_d):
    sigma_f = 48000
    return np.sqrt(np.exp((tau-tau_d)**2/sigma_f))

def eps(q_opt):
    return np.linalg.norm(q_opt, ord=2)**2

def L_5_loss(q_opt):
    L_5 = 0
    for s in range(n_srcs):
        L_5_ = 0
        q_opt_s = q_opt[s]
        for tau in range(J):
            L_5_ += ((1 - w_g(tau,0))*q_opt_s[tau]**2/eps(q_opt_s))**2
        L_5 += np.sqrt(L_5_)
    return L_5

def w_n(n,N, alpha=0.5):
    return 1+alpha*(n/N)


def L_6_loss(q_opt, rir):
    L_6 = 0
    
    for m in range(n_mics):
        L_6_ = 0
        h_tilde = 0
        for s in range(n_srcs):
            h_tilde += np.convolve(rir[m][s], q_opt[s])
        e = eps(h_tilde)
        for n in range(len(h_tilde)):
            L_6_ += ((1 - w_g(n, 1000))*h_tilde[n]**2/e)**2
        L_6 += np.sqrt(L_6_)
    return L_6

H, freqs = compute_H_matrix(NN_INPUT[-1][1])
H_B = torch.from_numpy(H[bright_zone_mics_index])  # Bright zone microphones
H_D = torch.from_numpy(H[dark_zone_mics_index])    # Dark zone microphones
M_B = len(bright_zone_mics_index)
M_D = len(dark_zone_mics_index)

H_time = compute_multi_toeplitz(NN_INPUT[-1][1], len(q[0]))

<<<<<<< HEAD:SZC_room_sim_NN_IR.py
#print("L_1", L_1_loss(q, fcentre, len(bright_zone_mics_index), H_B))
#print("L_2", L_2_loss(q))
#print("L_3", L_3_loss(q))
#print("L_4", L_4_loss(q))
#print("L_5", L_5_loss(q))
print("L_6", L_6_loss(q))
=======
print("L_1", L_1_loss(q, fcentre, M_B, H_B))
print("L_2", L_2_loss(q, fcentre, H_B, H_D, M_B, M_D))
print("L_3", L_3_loss(q, H_time))
print("L_4", L_4_loss(q))
print("L_5", L_5_loss(q))
print("L_6", L_6_loss(q, NN_INPUT[-1][1]))
>>>>>>> 7e198a5b22e8b2e638f3b9575759de48b18e2b4c:Junk/SZC_room_sim_NN_IR.py

exit()

def pressure_field_2d(room_dim, sources, q_opt, lyd_data, grid_res=50, z_plane=1.5, J=J, fs=16000):
    """
    Compute and plot a 2D grid of sound pressure (SPL) in the room at a fixed z-plane,
    using the speakers with the applied filters. Also prints average pressure in bright and dark zones.
    Accepts q_opt as either a torch tensor or numpy array.
    """
    L = len(sources)

    # Ensure q_opt is a numpy array of shape (L, J)
    if isinstance(q_opt, torch.Tensor):
        q_opt = q_opt.detach().cpu().numpy()
        if q_opt.ndim == 3:  # (batch, L, J)
            q_opt = q_opt[0]
    q_matrix = q_opt.reshape(L, J)

    # 2D grid at fixed z
    x = np.linspace(0, room_dim[0], grid_res)
    y = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, z_plane)

    pressure_field = np.zeros_like(X, dtype=float)
    test_signal = np.array(lyd_data[:fs//2])  # Use a short segment for speed

    for i in range(grid_res):
        for j in range(grid_res):
            point = [X[i, j], Y[i, j], Z[i, j]]
            h_point = []
            for src_idx, src_pos in enumerate(sources):
                r = np.linalg.norm(np.array(point) - np.array(src_pos))
                delay = int(r * fs / 343)
                h = np.zeros(J + 256)
                if delay < len(h):
                    h[delay] = 1 / (r + 1e-6)
                h_point.append(h)
            p = 0
            for l in range(L):
                filtered = lfilter(q_matrix[l], 1, test_signal)
                out_l = fftconvolve(filtered, h_point[l])
                p += np.sqrt(np.mean(out_l**2))  # RMS pressure
            pressure_field[i, j] = p

    pressure_dB = 20 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12) + 1e-12)

    center_x, center_y = vdg.spatial_positions[4][0], vdg.spatial_positions[4][1]
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    R_mic = vdg.R
    # Bright Zone: Inside or on the boundary of the microphone circle
    bright_mask = dist_sq <= (R_mic + 0.1)**2 # Added 0.1 buffer for visualization contrast
    dark_mask = dist_sq > (R_mic + 0.1)**2 
    
    avg_bright = np.mean(pressure_field[bright_mask])
    avg_dark = np.mean(pressure_field[dark_mask])
    
    print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
    contrast_db = 10.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12))
    print(f"Contrast (bright/dark) [dB] = {contrast_db:.2f}")


    plt.figure(figsize=(8, 6))
    plt.imshow(pressure_dB.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], aspect='auto', cmap='inferno')
        # Add Circular Bright/Dark zone markers
    theta = np.linspace(0, 2 * np.pi, 100)
    boundary_x = center_x + R_mic * np.cos(theta)
    boundary_y = center_y + R_mic * np.sin(theta)
    plt.plot(boundary_x, boundary_y, 'w--', linewidth=1, label='Zone Boundary')
    
    # Place text labels
    plt.text(center_x, center_y, 'Bright Zone', color='white', ha='center', fontsize=10, weight='bold')
    plt.text(center_x + R_mic + 0.2, center_y + R_mic + 0.2, 'Dark Zone', color='white', ha='left', fontsize=10, weight='bold')
    plt.colorbar(label='SPL [dB]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sound Pressure Level at z={z_plane} m')
    plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, label='Speakers')
    plt.legend()
    plt.tight_layout()
    plt.show()

def contrast_loss(pressure_field, X, Y, room_dim):
    """
    Loss function that maximizes the contrast between bright and dark zones.
    Bright zone: X < room_dim[0] / 2
    Dark zone:   X >= room_dim[0] / 2
    Returns negative contrast so optimizer maximizes it.
    """
    R_mic = vdg.R
    center_x, center_y = vdg.spatial_positions[4][0], vdg.spatial_positions[4][1]
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    bright_mask = dist_sq <= (R_mic + 0.1)**2 # Added 0.1 buffer for visualization contrast
    dark_mask = dist_sq > (R_mic + 0.1)**2 
    bright_mean = torch.mean(pressure_field[bright_mask])
    dark_mean = torch.mean(pressure_field[dark_mask])
    return -(bright_mean - dark_mean)

def compute_pressure_field_tensor(room_dim, sources, q_opt, lyd_data, grid_res=20, z_plane=1.5, J=J, fs=16000):

    """
    Differentiable PyTorch version: computes pressure field for given q_opt.
    Vectorized over grid points and sources for speed.
    Only direct-path, for demonstration.
    """
    device = q_opt.device
    L = len(sources)
    q_matrix = q_opt.view(L, J)

    # Grid points
    x = torch.linspace(0, room_dim[0], grid_res, device=device)
    y = torch.linspace(0, room_dim[1], grid_res, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.full_like(X, z_plane)
    grid_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (G, 3)
    G = grid_points.shape[0]

    src_positions = torch.tensor(sources, device=device)  # (L, 3)
    dists = torch.cdist(grid_points, src_positions)  # (G, L)
    delays = (dists * fs / 343).long()  # (G, L)

    # Build impulse responses: (G, L, J+256)
    h = torch.zeros(G, L, J + 256, device=device)
    idxs = torch.arange(G, device=device).unsqueeze(1).expand(-1, L)
    srcs = torch.arange(L, device=device).unsqueeze(0).expand(G, -1)
    valid = delays < h.shape[2]
    h[idxs[valid], srcs[valid], delays[valid]] = 1 / (dists[valid] + 1e-6)

    test_signal = torch.tensor(lyd_data[:fs//2], dtype=torch.float32, device=device)
    # Batch FIR filtering for all sources: (L, signal_len)
    filtered = torch.stack([
        F.conv1d(
            test_signal.view(1, 1, -1),
            q_matrix[l].view(1, 1, -1),
            padding=J-1
        ).view(-1)
        for l in range(L)
    ], dim=0)  # (L, signal_len)

    # Vectorized convolution for all grid points and sources
    # Prepare for broadcasting: (G, L, signal_len), (G, L, filter_len)
    filtered_exp = filtered.unsqueeze(0).expand(G, L, -1)  # (G, L, signal_len)
    h_exp = h  # (G, L, filter_len)

    # Flatten for batch processing
    filtered_flat = filtered_exp.reshape(G * L, -1).unsqueeze(1)  # [G*L, 1, signal_len]
    h_flat = h_exp.reshape(G * L, -1).unsqueeze(1)                # [G*L, 1, filter_len]

    # Transpose to [1, G*L, signal_len] for input and [G*L, 1, filter_len] for weight
    filtered_flat = filtered_flat.permute(1, 0, 2)  # [1, G*L, signal_len]

    out = F.conv1d(filtered_flat, h_flat, groups=G*L, padding=h.shape[2]-1)  # [1, G*L, output_len]
    out = out.permute(1, 0, 2).squeeze(1)  # [G*L, output_len]
    out = out.view(G, L, -1)

    # Compute RMS pressure for each grid point and sum over sources
    rms_pressure = torch.sqrt(torch.mean(out ** 2, dim=-1) + 1e-12)  # (G, L)
    pressure_field = torch.sum(rms_pressure, dim=1)  # (G,)
    pressure_field = pressure_field.view(grid_res, grid_res)
    return pressure_field, X, Y



    # Training loop - FIXED: use x_rir instead of x


"""rir = torch.Tensor(rir)

H, freqs = compute_H_matrix(rir_array)

H_B = torch.from_numpy(H[bright_zone_mics_index])  # Bright zone microphones

H_D = torch.from_numpy(H[dark_zone_mics_index])    # Dark zone microphones

H_time = compute_multi_toeplitz(rir, len(q[0]))

H_time = torch.Tensor(H_time)
H_time = H_time.to(q.dtype).detach()"""

num_epochs = 50
for epoch in range(num_epochs):
    for x_batch in NN_INPUT[:][0]:
        optimizer.zero_grad()

        # Forward pass
        q_batch = model(x_batch)  # shape: (batch_size, S, T)

        # Compute loss — you must define it batchwise
        loss = 0
        for b in range(q_batch.size(0)):
            q_opt = q_batch[b]
            #pressure_field, X, Y = compute_pressure_field_tensor(
            #    vdg.room_dim, sources_position_list, q_opt, lyd_data, grid_res=10, J=J, fs=fs
            #)
            loss += (2 * L_2_loss(q_opt)
                        + 126 * L_1_loss(q_opt)
                        + 1e-5 * L_3_loss(q_opt))
        loss /= q_batch.size(0)  # mean loss per batch

        if torch.isnan(loss):
            print(f"Epoch {epoch}, Loss: NaN (skipped update)")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}, Loss: {loss.item():.6f}")

# After training, visualize with the RIR-trained model
q_final = model(x_rir)[0]
pressure_field_2d(vdg.room_dim, sources_position_list, q_final.detach().cpu().numpy(), lyd_data, grid_res=50, z_plane=1.5, J=J, fs=fs)

# Save the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "SZC_model_rir.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully to: {model_path}")

def visualize_placement(room_dim, sources, mic_positions, bright_zone_mics, dark_zone_mics=None, center_line_mics=None):
    """
    Visualize the room layout with speakers and microphones
    
    Parameters:
    room_dim: list [L, W, H] - room dimensions
    sources: list of source positions
    mic_positions: numpy array of microphone positions (3 x N)
    bright_zone_mics: list of indices for bright zone microphones
    dark_zone_mics: list of indices for dark zone microphones (optional)
    center_line_mics: list of indices for center line microphones (optional)
    """
    
    fig = plt.figure(figsize=(15, 5))
    
    # Create 3 subplots: XY plane, XZ plane, and 3D view
    ax1 = fig.add_subplot(131)  # XY plane (top view)
    ax2 = fig.add_subplot(132)  # XZ plane (side view)
    ax3 = fig.add_subplot(133, projection='3d')  # 3D view
    
    # Plot room boundaries
    L, W, H = room_dim
    room_corners_xy = [[0, 0], [L, 0], [L, W], [0, W], [0, 0]]
    room_corners_xz = [[0, 0], [L, 0], [L, H], [0, H], [0, 0]]
    
    ax1.plot(*zip(*room_corners_xy), 'k-', linewidth=2, label='Room Boundary')
    ax2.plot(*zip(*room_corners_xz), 'k-', linewidth=2, label='Room Boundary')
    
    # Plot 3D room boundaries
    for z in [0, H]:
        ax3.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [z, z, z, z, z], 'k-', linewidth=1, alpha=0.5)
    for x in [0, L]:
        ax3.plot([x, x, x, x, x], [0, W, W, 0, 0], [0, 0, H, H, 0], 'k-', linewidth=1, alpha=0.5)
    for y in [0, W]:
        ax3.plot([0, L, L, 0, 0], [y, y, y, y, y], [0, 0, H, H, 0], 'k-', linewidth=1, alpha=0.5)
    
    # Plot sources (speakers)
    sources_array = np.array(sources)
    ax1.scatter(sources_array[:, 0], sources_array[:, 1], c='red', s=100, marker='^', label='Speakers', edgecolors='black')
    ax2.scatter(sources_array[:, 0], sources_array[:, 2], c='red', s=100, marker='^', edgecolors='black')
    ax3.scatter(sources_array[:, 0], sources_array[:, 1], sources_array[:, 2], c='red', s=100, marker='^', label='Speakers', edgecolors='black')
    
    # Plot microphones
    mic_array = mic_positions.T
    
    # Bright zone microphones
    bright_mics = mic_array[bright_zone_mics]
    ax1.scatter(bright_mics[:, 0], bright_mics[:, 1], c='blue', s=50, marker='o', label='Bright Zone Mics', alpha=0.8)
    ax2.scatter(bright_mics[:, 0], bright_mics[:, 2], c='blue', s=50, marker='o', alpha=0.8)
    ax3.scatter(bright_mics[:, 0], bright_mics[:, 1], bright_mics[:, 2], c='blue', s=50, marker='o', label='Bright Zone Mics', alpha=0.8)
    
    # Dark zone microphones (if provided)
    if dark_zone_mics is not None:
        dark_mics = mic_array[dark_zone_mics]
        ax1.scatter(dark_mics[:, 0], dark_mics[:, 1], c='orange', s=50, marker='s', label='Dark Zone Mics', alpha=0.8)
        ax2.scatter(dark_mics[:, 0], dark_mics[:, 2], c='orange', s=50, marker='s', alpha=0.8)
        ax3.scatter(dark_mics[:, 0], dark_mics[:, 1], dark_mics[:, 2], c='orange', s=50, marker='s', label='Dark Zone Mics', alpha=0.8)
    
    # Center line microphones (if provided)
    if center_line_mics is not None:
        center_mics = mic_array[center_line_mics]
        ax1.scatter(center_mics[:, 0], center_mics[:, 1], c='green', s=50, marker='D', label='Center Line Mics', alpha=0.8)
        ax2.scatter(center_mics[:, 0], center_mics[:, 2], c='green', s=50, marker='D', alpha=0.8)
        ax3.scatter(center_mics[:, 0], center_mics[:, 1], center_mics[:, 2], c='green', s=50, marker='D', label='Center Line Mics', alpha=0.8)
    
    # Add zone boundaries
    ax1.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='Zone Boundary')
    ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='Zone Boundary')
    ax2.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and titles
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY Plane)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D View')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary information
    print(f"Room dimensions: {room_dim[0]}m x {room_dim[1]}m x {room_dim[2]}m")
    print(f"Number of speakers: {len(sources)}")
    print(f"Number of microphones: {mic_positions.shape[1]}")
    if dark_zone_mics is not None:
        print(f"Bright zone microphones: {len(bright_zone_mics)}")
        print(f"Dark zone microphones: {len(dark_zone_mics)}")
    if center_line_mics is not None:
        print(f"Center line microphones: {len(center_line_mics)}")


visualize_placement(vdg.room_dim, sources_position_list, mic_positions, bright_zone_mics_index, dark_zone_mics_index)
