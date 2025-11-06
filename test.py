import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F
from tqdm import tqdm  # progressbar

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
    return 20 * np.log10(max_val / np.sqrt(mse))

def compute_CC(reference, measured):
    ref_mean = np.mean(reference)
    meas_mean = np.mean(measured)
    numerator = np.sum((reference - ref_mean) * (measured - meas_mean))
    denominator = np.sqrt(np.sum((reference - ref_mean)**2) * np.sum((measured - meas_mean)**2))
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
    n_mics, n_samples = unfiltered.shape
    n_srcs, filter_len = filters.shape
    filtered = np.zeros((n_mics, n_samples + filter_len - 1))
    for s in range(n_srcs):
        for m in range(n_mics):
            filtered[m] += convolve(unfiltered[m], filters[s], mode='full')
    filtered /= np.max(np.abs(filtered))
    return filtered

def acoustic_contrast(rir, filt, reference, bright_idx, dark_idx):
    p = compute_pressure_unfiltered(rir, reference)
    p_C = compute_pressure_filtered(filt, p.numpy())
    e_B = np.sum(p_C[bright_idx]**2)
    e_D = np.sum(p_C[dark_idx]**2)
    M_B = len(bright_idx)
    M_D = len(dark_idx)
    return (M_D / M_B) * (e_B / e_D) if e_D != 0 else 1e10

# -------------------------
# Performance evaluation fra .pt fil med reference og progressbar
# -------------------------

def performance_evaluation_pt(pt_path, reference_wav_path, bright_idx, dark_idx, save_path="average_performance.txt"):
    # Load reference
    fs, wav = wavfile.read(reference_wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    reference = wav / np.max(np.abs(wav))
    
    # Load data
    data = torch.load(pt_path)
    filters_list = data['selected_filters']
    X_test_list = data['X_test']
    rir_list = data.get('RIR', [None] * len(X_test_list))  # Optional

    # Metrics
    PESQ_B_list, PESQ_D_list = [], []
    STOI_B_list, STOI_D_list = [], []
    PSNR_B_list, PSNR_D_list = [], []
    CC_B_list, CC_D_list = [], []
    AC_list = []

    for filt, X_test, rir in tqdm(zip(filters_list, X_test_list, rir_list), total=len(X_test_list), desc="Evaluating"):
        filt_np = filt.numpy() if torch.is_tensor(filt) else np.array(filt)
        X_test_np = X_test.numpy() if torch.is_tensor(X_test) else np.array(X_test)
        
        bright_np = X_test_np[bright_idx].mean(axis=0)
        dark_np = X_test_np[dark_idx].mean(axis=0)
        
        # Trim reference til samme længde
        ref_len = min(len(reference), len(bright_np))
        ref_np = reference[:ref_len]
        bright_np = bright_np[:ref_len]
        dark_np = dark_np[:ref_len]

        # Metrics
        PESQ_B_list.append(compute_pesq_np(ref_np, bright_np))
        PESQ_D_list.append(compute_pesq_np(ref_np, dark_np))
        STOI_B_list.append(compute_STOI_np(ref_np, bright_np))
        STOI_D_list.append(compute_STOI_np(ref_np, dark_np))
        PSNR_B_list.append(compute_PSNR(ref_np, bright_np))
        PSNR_D_list.append(compute_PSNR(ref_np, dark_np))
        CC_B_list.append(compute_CC(ref_np, bright_np))
        CC_D_list.append(compute_CC(ref_np, dark_np))

        # Acoustic contrast hvis RIR findes
        if rir is not None:
            rir_tensor = rir if torch.is_tensor(rir) else torch.from_numpy(np.array(rir))
            reference_tensor = torch.from_numpy(ref_np).unsqueeze(0)
            AC_list.append(acoustic_contrast(rir_tensor, filt_np, reference_tensor, bright_idx, dark_idx))

    avg_metrics = {
        "PESQ_B": np.mean(PESQ_B_list),
        "PESQ_D": np.mean(PESQ_D_list),
        "STOI_B": np.mean(STOI_B_list),
        "STOI_D": np.mean(STOI_D_list),
        "PSNR_B": np.mean(PSNR_B_list),
        "PSNR_D": np.mean(PSNR_D_list),
        "CC_B": np.mean(CC_B_list),
        "CC_D": np.mean(CC_D_list),
    }

    if AC_list:
        avg_metrics["Acoustic_Contrast"] = np.mean(AC_list)

    with open(save_path, "w") as f:
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Gennemsnit gemt i '{save_path}'")
    return avg_metrics

# -------------------------
# Kør evaluering
# -------------------------
if __name__ == "__main__":
    pt_file = "random_selection_data.pt"
    reference_wav = "Performance Evaluation/reference.wav"
    bright_idx = [12]        # eksempel
    dark_idx = list(range(12))  # eksempel
    performance_evaluation_pt(pt_file, reference_wav, bright_idx, dark_idx)
