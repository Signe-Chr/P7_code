import torch
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------
# 1. Frequency-domain transfer matrices (Torch version)
# ----------------------------------------------------------
def compute_H_B_H_D_torch(rir_array, bright_zone_idx, dark_zone_idx, fs=16000, n_fft=None):
    n_mics, n_srcs, n_samples = rir_array.shape
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    H = torch.fft.rfft(rir_array, n=n_fft)
    freqs = torch.fft.rfftfreq(n_fft, 1 / fs)
    H_B = H[bright_zone_idx, :, :]
    H_D = H[dark_zone_idx, :, :]
    M_B = len(bright_zone_idx)
    M_D = len(dark_zone_idx)
    return H_B, H_D, M_B, M_D, freqs


# ----------------------------------------------------------
# 2. Acoustic contrast computation
# ----------------------------------------------------------
def AC_tilde(H_B, H_D, g, M_B, M_D):
    g = g.unsqueeze(-1)  # [n_srcs, 1]
    E_B = torch.sum(torch.abs(H_B @ g) ** 2)
    E_D = torch.sum(torch.abs(H_D @ g) ** 2) + 1e-12
    return (M_D / M_B) * (E_B / E_D)


# ----------------------------------------------------------
# 3. L_2 acoustic contrast loss (band-limited)
# ----------------------------------------------------------
def L_2_loss(q_opt, fcentres, H_B, H_D, M_B, M_D, fs=16000, AC_des=None):
    n_freqs = H_B.shape[-1]
    g_f = torch.fft.rfft(q_opt)  # [n_freqs]
    delta_f = fs / (2 * (n_freqs - 1))
    fd = 2 ** (1/6)
    total_loss = 0.0

    for f_c in fcentres:
        f_low, f_high = f_c / fd, f_c * fd
        k_low = int(np.ceil(float(f_low / delta_f)))
        k_high = min(n_freqs, int(np.ceil(float(f_high / delta_f))))

        band_loss = 0.0

        for k in range(k_low, k_high):
            AC_sim = AC_tilde(H_B[:, :, k], H_D[:, :, k], g_f[:H_B.shape[1]], M_B, M_D)
            # --- adaptive desired contrast ---
            if AC_des is None:
                AC_des_k = 10 ** (-50 / 10)  # fallback
            else:
                AC_des_k = AC_des  # use per-scene reference contrast

            w_AC = max((100 / f_c) ** 1.0, 1.0)  # emphasize low freqs
            C = F.relu(torch.real(AC_des_k * w_AC - AC_sim))
            band_loss += C ** 2

        total_loss += torch.sqrt(band_loss + 1e-12)
    return total_loss

def compute_pressure(rir, g):
    """
    Compute pressure at microphones via FFT-based linear convolution, fully vectorized.
    
    Parameters:
        rir: torch.Tensor [n_mics, n_srcs, n_samples]
        g: torch.Tensor [n_srcs, filter_len]
    
    Returns:
        p: torch.Tensor [n_mics, n_samples + filter_len - 1]
    """
    n_mics, n_srcs, n_samples = rir.shape
    filter_len = g.shape[1] if g.ndim > 1 else g.shape[0]
    output_len = n_samples + filter_len - 1
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))

    # FFT along the last dimension for RIR: [n_mics, n_srcs, n_fft]
    f_h = torch.fft.fft(torch.nn.functional.pad(rir, (0, n_fft - n_samples)), n=n_fft)
    # FFT of filters: [n_srcs, n_fft]
    f_g = torch.fft.fft(torch.nn.functional.pad(g, (0, n_fft - filter_len)), n=n_fft, dim=-1)

    # Multiply f_h * f_g across sources, sum over sources
    # f_h: [n_mics, n_srcs, n_fft], f_g: [n_srcs, n_fft] -> expand f_g
    f_g_exp = f_g.unsqueeze(0)            # [1, n_srcs, n_fft]
    p_f = torch.sum(f_h * f_g_exp, dim=1) # sum over sources -> [n_mics, n_fft]

    # Inverse FFT and truncate to output_len
    p = torch.fft.ifft(p_f).real[:, :output_len]  # [n_mics, output_len]
    return p
def compute_pressure_with_input(rir, g, x):
    """
    Compute microphone pressures including the input sound.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples] (torch.Tensor)
        g: [n_srcs, filter_len] (torch.Tensor)
        x: [n_srcs, n_input_samples] (torch.Tensor)

    Returns:
        p: [n_mics, n_output_samples] (torch.Tensor)
    """
    n_mics, n_srcs, n_rir = rir.shape
    _, filter_len = g.shape
    _, n_input = x.shape
    output_len = n_rir + filter_len + n_input - 2
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))

    # FFTs
    f_h = torch.fft.fft(torch.nn.functional.pad(rir, (0, n_fft - n_rir)), n=n_fft)
    f_g = torch.fft.fft(torch.nn.functional.pad(g, (0, n_fft - filter_len)), n=n_fft, dim=-1)
    f_x = torch.fft.fft(torch.nn.functional.pad(x, (0, n_fft - n_input)), n=n_fft, dim=-1)

    # Broadcast multiply and sum over sources
    f_gx = f_g * f_x              # [n_srcs, n_fft]
    f_gx_exp = f_gx.unsqueeze(0)  # [1, n_srcs, n_fft]
    p_f = torch.sum(f_h * f_gx_exp, dim=1)

    p = torch.fft.ifft(p_f).real[:, :output_len]
    return p

def AC_tilde_with_input(rir, g, x_input):
    """
    Compute acoustic contrast using real input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_samples] torch.Tensor
        g: [n_srcs, filter_len] torch.Tensor
        x_input: [n_srcs, n_input_samples] torch.Tensor

    Returns:
        AC: scalar
    """
    # Compute pressures at microphones
    p = compute_pressure_with_input(rir, g, x_input)  # [n_mics, n_output]
    
    # Split bright/dark zones
    # Assumes caller has selected appropriate rir slices for B/D
    return p

def AC_loss_with_input(rir, bright_idx, dark_idx, g, x_input):
    """
    Compute AC from bright/dark zones using input signal.
    
    Parameters:
        rir: [n_mics, n_srcs, n_samples]
        bright_idx: list or tensor of bright zone mic indices
        dark_idx: list or tensor of dark zone mic indices
        g: [n_srcs, filter_len]
        x_input: [n_srcs, n_input_samples]
    
    Returns:
        AC: scalar
    """
    p = compute_pressure_with_input(rir, g, x_input)  # [n_mics, n_out]
    p_B = p[bright_idx, :]
    p_D = p[dark_idx, :]
    
    E_B = torch.sum(torch.abs(p_B) ** 2)
    E_D = torch.sum(torch.abs(p_D) ** 2) + 1e-12
    M_B, M_D = len(bright_idx), len(dark_idx)
    return (M_D / M_B) * (E_B / E_D)

def L_2_loss_with_input(q_opt, fcentres, rir, bright_idx, dark_idx, x_input, fs=16000, AC_des=None):
    """
    Band-limited AC loss including input signal x_input.
    """
    n_mics, n_srcs, n_samples = rir.shape
    g_f = q_opt  # assuming time-domain filters
    delta_f = fs / n_samples
    fd = 2 ** (1/6)
    total_loss = 0.0

    for f_c in fcentres:
        f_low, f_high = f_c / fd, f_c * fd
        k_low = int(np.ceil(f_low / delta_f))
        k_high = min(n_samples, int(np.ceil(f_high / delta_f)))

        band_loss = 0.0
        for k in range(k_low, k_high):
            AC_sim = AC_loss_with_input(rir, bright_idx, dark_idx, g_f, x_input)
            
            # Adaptive desired contrast
            AC_des_k = AC_des if AC_des is not None else 10 ** (-50 / 10)
            
            w_AC = max((100 / f_c) ** 1.0, 1.0)
            C = F.relu(torch.real(AC_des_k * w_AC - AC_sim))
            band_loss += C ** 2

        total_loss += torch.sqrt(band_loss + 1e-12)

    return total_loss
