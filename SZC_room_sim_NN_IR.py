import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy.signal import lfilter
import os
import VAST_dictionary_generator as vdg
from mpl_toolkits.mplot3d import Axes3D
import VISUALIZE_q_matrix as vq
import torch.nn.functional as F


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Signe_sang.wav")
fs, lyd_data = wavfile.read(file_path)

lyd_data=list(np.array(lyd_data[0:fs*1])/max(lyd_data))


#plt.plot(lyd_data)
#plt.show()

print(f"Sample rate: {fs} Hz, length: {len(lyd_data):.2f}")

# ----------------------
# Room & array settings
# ----------------------

# Create a shoebox room
room = pra.ShoeBox(
    vdg.room_dim,
    fs=vdg.fs_target,
    materials=pra.Material(vdg.absorption),
    max_order=vdg.max_order,
)


sources_position_list, mic_positions, bright_zone_mics_index, dark_zone_mics_index = vdg.sources_mics(vdg.R, vdg.spatial_positions[4], 12)


room.add_microphone_array(pra.MicrophoneArray(np.array(mic_positions).T, room.fs))

for s in sources_position_list:
        room.add_source(s)

# ----------------------
# Compute RIRs
# ----------------------
room.compute_rir()  # fills room.rir [mic_index][source_index] -> array

n_mics = len(mic_positions)
n_srcs = len(sources_position_list)

IR = room.rir

def compute_H_matrix(room, n_fft=None):
    """
    Compute the frequency-domain transfer matrix H[k]
    from the room impulse responses (RIRs) in a pyroomacoustics simulation.

    Parameters
    ----------
    room : pra.ShoeBox
        A pyroomacoustics room after calling room.compute_rir().
        room.rir[m][s] must contain the RIR from source s to mic m.
    n_fft : int, optional
        FFT length. If None, it uses the next power of 2 greater than 
        the longest RIR length. Defaults to 1024 if no RIRs are found.

    Returns
    -------
    H : np.ndarray, shape (n_mics, n_srcs, n_freqs)
        Frequency response matrix for all microphone–source pairs.
    freqs : np.ndarray
        Frequency vector corresponding to the frequency bins.
    """

    # We must check if the room RIRs have actually been computed
    if not hasattr(room, 'rir') or not room.rir:
        print("Warning: room.rir is empty. Ensure room.compute_rir() was called successfully.")
        # Fallback to a safe FFT size and empty results
        n_fft = n_fft if n_fft is not None else 1024
        freqs = np.fft.rfftfreq(n_fft, 1 / room.fs)
        return np.zeros((0, 0, len(freqs)), dtype=np.complex128), freqs

    n_mics = len(room.mic_array.R[0]) if room.mic_array is not None else 0
    n_srcs = len(room.sources)

    # --- CORRECTION APPLIED HERE ---
    # Find max RIR length across all mic–source pairs safely.
    # The 'or [0]' ensures max() always has at least one element.
    all_rir_lengths = [len(rir) for mic_rirs in room.rir for rir in mic_rirs]
    max_len = max(all_rir_lengths) if all_rir_lengths else 0
    
    # If n_fft is not specified, calculate it
    if n_fft is None:
        if max_len == 0:
            n_fft = 1024  # Default FFT length if no RIRs were found
        else:
            # Use the next power of 2 for efficient FFT
            n_fft = 2 ** int(np.ceil(np.log2(max_len)))

    n_freqs = n_fft // 2 + 1
    
    # Initialize frequency-domain matrix
    H = np.zeros((n_mics, n_srcs, n_freqs), dtype=np.complex128)

    # Compute FFT for each microphone–source pair
    for m in range(n_mics):
        for s in range(n_srcs):
            # Check if the RIR list for this pair exists and is not empty
            if m < len(room.rir) and s < len(room.rir[m]):
                h = np.array(room.rir[m][s])
                if len(h) > 0:
                    # Use rfft which only computes the first half of the spectrum
                    H[m, s, :] = np.fft.rfft(h, n=n_fft)

    freqs = np.fft.rfftfreq(n_fft, 1 / room.fs)
    return H, freqs

H, freqs = compute_H_matrix(room)

H_B = H[bright_zone_mics_index]  # Bright zone microphones

H_D = H[dark_zone_mics_index]    # Dark zone microphones

def compute_H_B_time(room, bright_zone_mics):
    """
    Compute the time-domain spatial transfer matrix H_B[n]
    for the bright zone microphones in a pyroomacoustics simulation.

    Parameters
    ----------
    room : pra.ShoeBox
        A pyroomacoustics room after calling room.compute_rir().
        room.rir[m][s] is the RIR from source s to mic m.
    bright_zone_mics : list of int
        Indices of microphones that belong to the bright zone.

    Returns
    -------
    H_B : np.ndarray
        Time-domain matrix of shape (n_samples, M_B, L)
        Each entry H_B[t, m, l] = h_{m,l}[t], the RIR sample at time t.
    t : np.ndarray
        Time vector in seconds corresponding to the samples.
    """

    if not hasattr(room, 'rir') or not room.rir:
        raise ValueError("room.rir is empty. Ensure room.compute_rir() was called successfully.")

    n_srcs = len(room.sources)
    n_mics_total = len(room.rir)
    M_B = len(bright_zone_mics)

    # Determine maximum RIR length
    max_len = max(len(r) for mic_rirs in room.rir for r in mic_rirs if len(r) > 0)
    
    # Initialize H_B
    H_B = np.zeros((max_len, M_B, n_srcs))

    # Fill in the impulse responses
    for i_bz, m in enumerate(bright_zone_mics):
        for s in range(n_srcs):
            if m < len(room.rir) and s < len(room.rir[m]):
                h = np.array(room.rir[m][s])
                H_B[:len(h), i_bz, s] = h  # zero-pad as needed

    t = np.arange(max_len) / room.fs
    return H_B, t    

H_B_time, t_b = compute_H_B_time(room, bright_zone_mics_index)

H_D_time, t_d = compute_H_B_time(room, dark_zone_mics_index)


# ----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


J = vdg.J


# Convert RIRs to a suitable tensor format for CNN input
def prepare_rir_input(IR, n_mics, n_srcs, max_length=512):
    """
    Prepare RIR data as CNN input tensor
    Shape: (batch_size, channels, n_mics, n_srcs, time)
    """
    # Create a tensor to hold all RIRs
    rir_tensor = torch.zeros(1, 1, n_mics, n_srcs, max_length)
    rir_list = []

    for mic_idx in range(n_mics):
        rir_temp = []
        for src_idx in range(n_srcs):
            rir = IR[mic_idx][src_idx]
            # Truncate or zero-pad to max_length
            if len(rir) > max_length:
                rir = rir[:max_length]
            else:
                rir = np.pad(rir, (0, max_length - len(rir)))
            rir_tensor[0, 0, mic_idx, src_idx, :] = torch.tensor(rir)
            rir_temp.append(rir)
        rir_list.append(rir_temp)

    
    return rir_tensor, np.array(rir_list)

# Prepare the input tensor
x_rir, rir = prepare_rir_input(IR, n_mics, n_srcs, max_length=J)




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


model = ILZ_CNN_RIR(M=n_mics, S=n_srcs, T=J, K=J)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)  # Lower learning rate
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) #lr: learning rate, weight_decay: L2 regularization
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

q = vq.q[0]
fcentre = [1000, 2000]

def L_1_loss(q_opt):
    g = np.fft.fft(q_opt, axis = 0)
    target_pressure = np.abs(g)*1.3 # revurderes
    L_1 = 0

    for freq in fcentre:
        fd = 2**(1/6)
        f_low = freq/fd
        f_high = freq*fd
        delta_f = vdg.fs_target/vdg.J

        k_low = int(np.ceil(f_low/delta_f))

        k_high = int(np.ceil(f_high/delta_f))

        L_1_ = 0

        for m in range(len(bright_zone_mics_index)):
            temp_1 = 0
            for k in range(k_low, k_high):
                H_B_tilde = np.matmul(H_B[:,:,k], g)
                temp_1 += (np.linalg.norm(H_B_tilde[m,:], ord=1)-np.linalg.norm(target_pressure[:,k], ord=1))**2
            L_1_ += np.sqrt(temp_1)
        L_1 += L_1_
    return L_1

def C_i(AC_des, w_AC, AC_tilde):
    #print(np.real(AC_des * w_AC - AC_tilde))
    return max(0, np.real(AC_des * w_AC - AC_tilde))
    
def w_ac(center_frequencies: list, ref_frequency: float = 100.0, 
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
    f_i = np.array(center_frequencies)
    
    # Calculate the ratio raised to the power beta
    weight_ratios = (ref_frequency / f_i) ** beta
    
    # Ensure the weight never drops below the specified minimum weight
    w_ac = np.maximum(weight_ratios, min_weight)
    
    return w_ac.tolist()

def AC_tilde(H_B, H_D, g):
    return (len(dark_zone_mics_index)*(g.H*H_B.H)*(H_B*g))/(len(bright_zone_mics_index)*((g.H*H_D.H)*(H_D*g)))

def L_2_loss(q_opt):
    L_2 = 0
    for freq in fcentre:
        fd = 2**(1/6)
        f_low = freq/fd
        f_high = freq*fd
        delta_f = vdg.fs_target/vdg.J

        k_low = int(np.ceil(f_low/delta_f))

        k_high = int(np.ceil(f_high/delta_f))
        L_2_ = 0
        for i in range(k_low, k_high):
            g = np.matrix(np.fft.fft(q_opt, axis = 0))
            AC_sim = AC_tilde(np.matrix(H_B[:,:,i]), np.matrix(H_D[:,:,i]), g[:,i])
            AC_des = 10**(-50/10)#5.079192938063992e-07
            w_AC = w_ac([freq], ref_frequency=100.0, beta=1.0, min_weight=1.0)[0]
            C = C_i(AC_des, w_AC, AC_sim)
            L_2_ += C**2
        L_2 += np.sqrt(L_2_)
    return L_2

def energy_tilde(q_opt, mic_index, speaker_index):
    e_b = 0
    for i in range(vdg.N):
        e_b += (np.matrix(q_opt)[speaker_index]*H_B_time[i,mic_index,speaker_index])*(H_B_time[i,mic_index,speaker_index]*np.matrix(q_opt)[speaker_index].T)
    return e_b[0,0]

def energy(mic_index, speaker_index):
    e_b = 0
    for i in range(vdg.N):
        e_b += (H_B_time[i,mic_index,speaker_index])*(H_B_time[i,mic_index,speaker_index])
    return e_b

def L_3_loss(q_opt):
    L_3 = 0
    for m in range(len(bright_zone_mics_index)):
        mm = 0
        for s in range(n_srcs):
            mm += (energy_tilde(q_opt, m, s)/energy_tilde(q_opt, m, -1) - energy(m, s)/energy(m, -1))**2
        L_3 += np.sqrt(mm)

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


def L_6_loss(q_opt):
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

print(L_6_loss(q), L_5_loss(q), L_4_loss(q), L_3_loss(q), L_2_loss(q), L_1_loss(q))


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
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Use the RIR data as input
    q_opt = model(x_rir)[0]  # shape: (S, J)
    
    pressure_field, X, Y = compute_pressure_field_tensor(vdg.room_dim, sources_position_list, q_opt, lyd_data, grid_res=10, J=J, fs=fs)
    
    loss = 2*L_6_loss(q_opt.detach()) + 92*L_5_loss(q_opt.detach()) + 50*L_4_loss(q_opt.detach()) + 1e-5*L_3_loss(q_opt.detach()) + 2*L_2_loss(q_opt.detach()) + 126*L_1_loss(q_opt.detach())# 
    
#100*contrast_loss(pressure_field, X, Y, vdg.room_dim) + 
    if torch.isnan(loss):
        print(f"Epoch {epoch}, Loss: NaN (skipped update)")
        continue
        
    loss.backward()
    
    # Gradient clipping to prevent explosions
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
