import torch
import torch.nn.functional as F
import Dataset_generator_script as dgs

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
        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(n_samples))))  # next power of 2

    n_freqs = n_fft // 2 + 1

    # --- Allocate frequency-domain matrix ---
    H = torch.zeros((n_mics, n_srcs, n_freqs), dtype=torch.complex128)

    # --- Compute FFT for each mic–source pair ---
    for m in range(n_mics):
        for s in range(n_srcs):
            h = rir_array[m, s, :]
            H[m, s, :] = torch.fft.rfft(h, n=n_fft)

    # --- Frequency axis ---
    freqs = torch.fft.rfftfreq(n_fft, 1 / fs)

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
    output_len = torch.tensor(n_rir_samples + filter_len + n_input_samples - 2)
    
    p = torch.zeros((n_mics, output_len), device=rir.device)

    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            # Combined filter impulse response: h_combined = RIR * filter_q (via standard convolution)
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0).float()  # cast to float32
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0).float()    # cast to float32

            h_combined = F.conv1d(rir_m_s, q_s, padding=q_s.shape[-1]-1)
            
            p_m_s = F.conv1d(x_input.reshape((1, 1, n_input_samples)).float(), h_combined, padding=h_combined.shape[-1]-1).squeeze()
            
            p_m += p_m_s
        p[m, :] = p_m
    return p

MSE = torch.nn.MSELoss()

def Cosine_similarity(predicted_filter: torch.Tensor, true_filter: torch.Tensor):
    """Cosine distance between two flattened filters."""
    y_test_norm = F.normalize(predicted_filter, p=2, dim=1)
    y_cand_norm = F.normalize(true_filter, p=2, dim=1)
    similarity = torch.mm(y_test_norm, y_cand_norm.T)
    cosine_distance = 1 - similarity.squeeze()
    return cosine_distance

def MSEP(predicted_filter: torch.Tensor, true_filter: torch.Tensor,
         rir_test: torch.Tensor, x_input: torch.Tensor,
         B_idx: list, D_idx: list) -> torch.Tensor:
    """
    Compute MSPE (Mean Squared Pressure Error) only in the Bright Zone (B_idx)
    between the desired pressure (from test filter/RIR) and the predicted pressure 
    (from candidate filter/train RIR).
    """
    
    # 1. Calculate Desired Pressure (Reference: Test Filter + Test RIR)
    p_des_full = compute_pressure_with_input(rir_test, true_filter, x_input) # [n_mics, n_samples]
    p_des_B = p_des_full[B_idx] # [M_B, n_samples]
    p_des_D = p_des_full[D_idx]

    # 2. Calculate Predicted Pressure (Candidate: Candidate Filter + Train RIR)
    p_pred_full = compute_pressure_with_input(rir_test, predicted_filter, x_input) # [n_mics, n_samples]
    p_pred_B = p_pred_full[B_idx] # [M_B, n_samples]
    p_pred_D = p_pred_full[D_idx]

    # 3. Compute MSE
    msep_loss_B = torch.mean((p_pred_B - p_des_B) ** 2)
    msep_loss_D = torch.mean((p_pred_D - p_des_D) ** 2)
    return msep_loss_B, msep_loss_D

def AC_loss(q_pred, q_true, H, bright_indices, dark_indices):
    M_B = len(bright_indices)
    M_D = len(dark_indices)
    fcentres = torch.tensor([1000, 2000])
    fd = torch.tensor(2**(1/6))
    delta_f = dgs.fs_target/dgs.J
    g = torch.fft.fft(q_pred, axis = 0)
    g_true = torch.fft.fft(q_true, axis = 0)
    AC_loss_total = 0
    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd
        k_low = int(torch.ceil(f_low/delta_f))
        k_high = int(torch.ceil(f_high/delta_f))
        AC_loss_temp = torch.tensor(0,dtype=torch.float32)
        for k in range(k_low, k_high):
            AC_des = AC_tilde(H[bright_indices][:,:,k], H[dark_indices][:,:,k], g_true[:,k], M_B, M_D)
            AC_sim = AC_tilde(H[bright_indices][:,:,k], H[dark_indices][:,:,k], g[:,k], M_B, M_D)
            w_AC = w_ac(freq, ref_frequency=100, beta=1, min_weight=1)
            C = C_i(AC_des, w_AC, AC_sim)
            if torch.isnan(C):
                continue
            AC_loss_temp += C**2
        AC_loss_total += torch.sqrt(AC_loss_temp)
    return AC_loss_total


import numpy as np
def L_1_reg(q_pred, q_true, H, bright_indices):
    g = torch.fft.fft(q_pred, axis = 0)
    g_true = torch.fft.fft(q_true, axis = 0)
    loss = 0
    N = len(g_true)
    for n in range(N):
        p_true_abs = torch.abs(torch.matmul(H[bright_indices][:,:,n].to(torch.complex128), g_true[:,n].to(torch.complex128)))
        p_abs = torch.abs(torch.matmul(H[bright_indices][:,:,n].to(torch.complex128), g[:,n].to(torch.complex128)))
        loss = loss + torch.abs((p_true_abs - p_abs).pow(2))
    return 1/(N*len(bright_indices)) * loss
        
def L_2_reg(q_pred, H, dark_indices):
    M_D = len(dark_indices)
    g = torch.fft.fft(q_pred, axis = 0)
    N = len(g)
    total_loss=0
    for n in range(N):
        total_loss+torch.abs(torch.matmul(H[dark_indices][:,:,n].to(torch.complex128), g[:,n].to(torch.complex128))).pow(2)
    return(1/(N*M_D)*total_loss)

def L_3_reg(q_pred, L, g_max=1):
    N = len(q_pred)

    # FFT along the filter length
    g = torch.fft.fft(q_pred, dim=1)   # q_pred is already (3,1024)

    # magnitude spectrum
    mag = torch.abs(g)                 # shape (3,N)

    # maximum allowed magnitude per loudspeaker (vector of 3 identical values)
    bound = g_max * torch.ones(3, device=q_pred.device)

    total_loss = 0.0

    for n in range(N):
        # max(0, |g| - bound)
        excess = torch.relu(mag[:, n] - bound)

        # squared norm
        total_loss += torch.sum(excess**2)

    return total_loss / (N * L)


from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4, N=8192):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Filter coefficients
    b, a = butter(order, [low, high], btype='band')

    # Create unit impulse
    impulse = np.zeros(N)
    impulse[0] = 1.0

    # Filter the impulse to obtain impulse response
    h = filtfilt(b, a, impulse)
    return h

import torch.nn.functional as F
def L_4_reg(q_pred, dev):
    L = np.shape(q_pred)[0]
    N_hat = len(q_pred)
    f = torch.from_numpy(butter_bandpass(100, 1500, fs=8192).copy())
    q_temp = q_pred[0]
    N = np.shape(q_temp)[0] + np.shape(f)[0] - 1
    w = torch.from_numpy(1.0 - np.hamming(N)).to(dev)
    total_loss = 0.0
    for l in range(L):
        #conv_out = F.conv1d(input = q_pred[l].view(1, 1, -1).to(torch.float).to(dev), weight=f.view(1, 1, -1).to(torch.float).to(dev), padding='same').to(dev)
        padding = q_pred[l].shape[-1] - 1
        conv_out = F.conv1d(input = f.view(1, 1, -1).to(torch.float).to(dev), weight=q_pred[l].view(1, 1, -1).to(torch.float).to(dev), padding = padding).to(dev)
        weighted = w * conv_out
        total_loss += torch.sum(weighted**2)
    return (1/(N_hat * L)) * total_loss

        
