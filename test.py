import os
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample, convolve
from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F

# -------------------------
# Hjælpefunktioner
# -------------------------

def compute_pesq_np(reference, measured):
    return pesq(16000, reference, measured, 'wb')

def compute_STOI_np(reference, measured):
    return stoi(reference, measured, 16000, extended=False)

def compute_PSNR(reference, measured):
    mse = np.mean((reference - measured)**2)
    if mse == 0:
        return np.inf
    max_val = np.max(np.abs(reference))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def compute_CC(reference, measured):
    reference_mean = np.mean(reference)
    measured_mean = np.mean(measured)
    numerator = np.sum((reference - reference_mean) * (measured - measured_mean))
    denominator = np.sqrt(np.sum((reference - reference_mean)**2) * np.sum((measured - measured_mean)**2))
    return numerator / denominator if denominator != 0 else 0

def compute_pressure_unfiltered(rir: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    n_mics, n_srcs, n_rir_samples = rir.shape
    n_input_samples = reference.shape[-1]
    output_len = n_rir_samples + n_input_samples - 1
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len))
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))
    X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)
    for m in range(n_mics):
        p_m = torch.zeros(output_len)
        for s in range(n_srcs):
            h = rir[m, s, :]
            h_padded = F.pad(h, (0, output_len - n_rir_samples), 'constant', 0)
            H_fft = torch.fft.rfft(h_padded, n=n_fft)
            P_fft = H_fft * X_fft
            p_m += torch.fft.irfft(P_fft, n=n_fft)[:output_len]
        p[m, :] = p_m
    return p

def compute_pressure_filtered(filters, unfiltered):
    """
    Convolve each source with its filter and sum across sources.
    Keeps all microphones.
    
    filters: [n_srcs, filter_len]
    unfiltered: [n_mics, n_samples]
    """
    n_mics, n_samples = unfiltered.shape
    n_srcs, filter_len = filters.shape
    filtered = np.zeros((n_mics, n_samples + filter_len - 1))
    
    for s in range(n_srcs):
        for m in range(n_mics):
            filtered[m] += convolve(unfiltered[m], filters[s], mode='full')
    
    # Normaliser
    filtered /= np.max(np.abs(filtered))
    return filtered

def acoustic_contrast(rir, filt, reference, bright_idx, dark_idx):
    p = compute_pressure_unfiltered(rir, reference)
    p_C = compute_pressure_filtered(filt, p.numpy())
    e_B = np.sum(p_C[bright_idx]**2)
    e_D = np.sum(p_C[dark_idx]**2)
    M_B = len(bright_idx)
    M_D = len(dark_idx)
    AC = (M_D / M_B) * (e_B / e_D) if e_D != 0 else 1e10
    return AC

# -------------------------
# Performance evaluation for alle filer
# -------------------------

def performance_evaluation_all(folder_path, reference_wav_path, save_path="average_performance.txt"):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    
    PESQ_B_list, PESQ_D_list = [], []
    STOI_B_list, STOI_D_list = [], []
    PSNR_B_list, PSNR_D_list = [], []
    CC_B_list, CC_D_list = [], []
    AC_list = []

    # Load reference audio
    fs, wav = wavfile.read(reference_wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav / np.max(np.abs(wav))
    reference = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        data_dict = np.load(filepath, allow_pickle=True).item()

        IR = torch.from_numpy(data_dict['IR']).float()           # [n_mics, n_srcs, n_rir_samples]
        filters = data_dict['q_matrix'].astype(np.float32)       # [n_srcs, filter_len]
        bright_idx = data_dict['bright_zone_mics_index']        # fx [12]
        dark_idx = data_dict['dark_zone_mics_index']            # fx [0..11]

        # Simuler tryk
        unfiltered = compute_pressure_unfiltered(IR, reference)
        filtered = compute_pressure_filtered(filters, unfiltered.numpy())

        # Bright / Dark zone
        bright_np = filtered[bright_idx].mean(axis=0)
        dark_np   = filtered[dark_idx].mean(axis=0)
        ref_np = reference.squeeze().numpy()
        bright_np = bright_np[:len(ref_np)]
        dark_np   = dark_np[:len(ref_np)]

        # Metrics
        PESQ_B_list.append(compute_pesq_np(ref_np, bright_np))
        PESQ_D_list.append(compute_pesq_np(ref_np, dark_np))
        STOI_B_list.append(compute_STOI_np(ref_np, bright_np))
        STOI_D_list.append(compute_STOI_np(ref_np, dark_np))
        PSNR_B_list.append(compute_PSNR(ref_np, bright_np))
        PSNR_D_list.append(compute_PSNR(ref_np, dark_np))
        CC_B_list.append(compute_CC(ref_np, bright_np))
        CC_D_list.append(compute_CC(ref_np, dark_np))
        AC_list.append(acoustic_contrast(IR, filters, reference, bright_idx, dark_idx))

    # Gennemsnit
    avg_metrics = {
        "PESQ_B": np.mean(PESQ_B_list),
        "PESQ_D": np.mean(PESQ_D_list),
        "STOI_B": np.mean(STOI_B_list),
        "STOI_D": np.mean(STOI_D_list),
        "PSNR_B": np.mean(PSNR_B_list),
        "PSNR_D": np.mean(PSNR_D_list),
        "CC_B": np.mean(CC_B_list),
        "CC_D": np.mean(CC_D_list),
        "Acoustic_Contrast": np.mean(AC_list)
    }

    # Gem til fil
    with open(save_path, "w") as f:
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Gennemsnit gemt i '{save_path}'")
    return avg_metrics

# -------------------------
# Kør evaluering
# -------------------------
if __name__ == "__main__":
    folder = "Signes_data"
    reference_wav = "relaxing-guitar-loop-v5-245859.wav"
    performance_evaluation_all(folder, reference_wav)

