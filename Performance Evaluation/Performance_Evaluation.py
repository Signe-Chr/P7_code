import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F
from tqdm import tqdm

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
    return (M_D / M_B) * (e_B / e_D) if e_D != 0 else 1e10

# -------------------------
# Performance evaluation
# -------------------------

def performance_evaluation_pt(pt_path, reference_wav_path, rir_tensor, bright_idx, dark_idx, save_path="performance.txt"):
    # Load reference
    fs, wav = wavfile.read(reference_wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    reference = wav / np.max(np.abs(wav))
    reference_tensor = torch.from_numpy(reference.astype(np.float32)).unsqueeze(0)

        # Load .pt data
    data = torch.load(pt_path)

    # Understøt både gamle og nye formater
    if isinstance(data, dict) and 'selected_filters' in data:
        filters_list = data['selected_filters']  # [n_trials, filter_len_total]
        print(f"Indlæste {len(filters_list)} filtre fra gammel struktur ({pt_path})")
    elif isinstance(data, dict):
        # Ny struktur: dict med {filename: predicted_filter_tensor}
        filters_list = list(data.values())
        print(f"Indlæste {len(filters_list)} filtre fra regressionstruktur ({pt_path})")
        first_key = list(data.keys())[0]
        print(f"Eksempel: {first_key} -> shape {data[first_key].shape}")
    else:
        raise ValueError(f"Ukendt dataformat i {pt_path}")
    
    # Metrics
    PESQ_B_list, PESQ_D_list = [], []
    STOI_B_list, STOI_D_list = [], []
    PSNR_B_list, PSNR_D_list = [], []
    CC_B_list, CC_D_list = [], []
    AC_list = []

    # Compute unfiltered pressure
    unfiltered = compute_pressure_unfiltered(rir_tensor, reference_tensor).numpy()
    
    for filt_flat in tqdm(filters_list, total=len(filters_list), desc="Evaluating"):
        # Reshape filter til [n_srcs=3, filter_len]
        filt_np = filt_flat.numpy().reshape(3, -1) if torch.is_tensor(filt_flat) else np.array(filt_flat).reshape(3, -1)

        

        # Apply filters
        filtered = compute_pressure_filtered(filt_np, unfiltered)

        # Bright / Dark zones
        bright_np = np.atleast_1d(filtered[bright_idx].mean(axis=0))
        dark_np   = np.atleast_1d(filtered[dark_idx].mean(axis=0))

        # Trim reference
        ref_len = min(len(reference), len(bright_np))
        ref_np = reference[:ref_len]
        bright_np = bright_np[:ref_len]
        dark_np   = dark_np[:ref_len]

        # Metrics
        PESQ_B_list.append(compute_pesq_np(ref_np, bright_np))
        PESQ_D_list.append(compute_pesq_np(ref_np, dark_np))
        STOI_B_list.append(compute_STOI_np(ref_np, bright_np))
        STOI_D_list.append(compute_STOI_np(ref_np, dark_np))
        PSNR_B_list.append(compute_PSNR(ref_np, bright_np))
        PSNR_D_list.append(compute_PSNR(ref_np, dark_np))
        CC_B_list.append(compute_CC(ref_np, bright_np))
        CC_D_list.append(compute_CC(ref_np, dark_np))
        AC_list.append(acoustic_contrast(rir_tensor, filt_np, reference_tensor, bright_idx, dark_idx))

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
    
    file_names = ("Saved Filters/regression_filters.pt", "Saved Filters/random_selection_data.pt")
    pt_file = file_names[0]  # vælg hvilken .pt fil der skal evalueres
    reference_wav = "Performance Evaluation/reference.wav"
    
    # RIR skal være tensor med shape [n_mics, n_srcs, n_rir_samples]
    # Fx load fra .npy fil: rir_tensor = torch.from_numpy(np.load("rir.npy")).float()
    #rir_tensor = torch.load("Performance Evaluation/test_ir.pt",weights_only=False)  # erstat med din RIR

    n_mics = 13
    n_srcs = 3
    n_rir_samples = 1024  # fx
    rir_tensor = torch.randn(n_mics, n_srcs, n_rir_samples)

    bright_idx = [0]               # 1 bright zone mikrofon
    dark_idx = list(range(1, 13))  # 12 dark zone mikrofoner

    performance_evaluation_pt(pt_file, reference_wav, rir_tensor, bright_idx, dark_idx)
