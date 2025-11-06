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
            
            n_fft = 2**int(torch.ceil(torch.log2(output_len)))
            
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

def L_4_loss(q_true, q_pred, rir, x_input, H, bright_indices, dark_indices):
    M_B = len(bright_indices)
    M_D = len(dark_indices)
    fcentres = torch.tensor([1000, 2000])
    fd = torch.tensor(2**(1/6))
    delta_f = dgs.fs_target/dgs.J
    L_4 = 0
    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd
        g = torch.fft.fft(q_pred, axis = 0)
        p_des_full = compute_pressure_with_input(rir, q_true, x_input)
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

