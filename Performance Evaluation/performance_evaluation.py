import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import convolve, stft, resample
from scipy.io import loadmat, wavfile
from pesq import pesq
import torch.nn.functional as F
from pystoi import stoi
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
from MLP_classification import SoftFilterNet
from VAST_filter_coefficients import setup_acoustic_scenario
from Dataset_generator_script import sources_mics, fs_target, rooms #, #mic_directions, mic_positions_list, bright_zone_mics_index


def resample_to_16k_np(wav_path):
    fs, audio = wavfile.read(wav_path)
    # convert to float32 in [-1,1]
    if audio.dtype != np.float32:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))
    if fs != 16000:
        n_samples = int(len(audio) * 16000 / fs)
        audio = resample(audio, n_samples)
    return audio

#---Load data---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "Signes_data"
full_data = os.listdir(data_dir)
data_points = []
for data in full_data:
    if int(data.split("_")[1]) in ri:
        data_points.append(data)
data = CustomDataset(data_dir,data_points)
data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
Q = [batch for batch in data_loader][0]
X = Q[0]
filters = Q[1]
bright_zone_mics_index = Q[2]
dark_zone_mics_index = Q[3]
n_srcs = Q[4]
RIRs = Q[5]


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
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
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
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(x_input_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p
#print(compute_pressure_with_input(RIRs[0].to(device),filters[0].to(device),torch.randn(1,16000).to(device)).shape)

# -------------------------------------------------------------------------
# 3. Performance Evaluation
# -------------------------------------------------------------------------
''' Other metrics:
    MSE or RMSE
    Correlation, CC
    RSRQ (SNR lignende)
    SNR, PSNR eller PSIR (peak-snr eller sir)
    Log-Spectral Distance (LSD)
    Cosine similarity
'''

def compute_pesq(original, measured, mode:str='wb'):
    """
    Calculate PESQ (MOS-LQO) score between a original and measured audio file.

    Args:
        reference_file (str): Path to the reference WAV file.
        degraded_file (str): Path to the degraded WAV file.
        mode (str): 'nb' for narrowband (8 kHz) or 'wb' for wideband (16 kHz). Default is 'wb'.

    Returns:
        float: PESQ score (MOS-LQO) ranging approximately from 1.0 to 4.5.
    """
    fs_ref, ref = wavfile.read(original)
    fs_deg, deg = wavfile.read(measured)

    # Check that sample rates match
    if fs_ref != fs_deg:
        raise ValueError("Sample rates of reference and degraded files do not match.")

    # Make sure the signals are mono (1D arrays)
    if ref.ndim > 1:
        ref = ref[:, 0]
    if deg.ndim > 1:
        deg = deg[:, 0]

    # Trim or pad signals to the same length
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    ref = resample_to_16k_np(original)
    deg = resample_to_16k_np(measured)
    score = pesq(16000, ref, deg, mode)
    # Calculate PESQ score
    #mos = pesq.mos(ref, deg)
    return score

def compute_psnr(original, measured):
    mse = np.mean((original - measured)**2)
    if mse == 0:
        return np.inf
    max_val = np.max(np.abs(original))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def compute_lsd(original, measured, fs, n_fft=1024, hop_length=512):
    # Short-Time Fourier Transform
    f1, t1, Z_orig = stft(original, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    f2, t2, Z_meas = stft(measured, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Power spectrum in dB
    S_orig = 20 * np.log10(np.abs(Z_orig) + 1e-10)
    S_meas = 20 * np.log10(np.abs(Z_meas) + 1e-10)
    
    # LSD per frame
    lsd_frames = np.sqrt(np.mean((S_orig - S_meas)**2, axis=0))
    return np.mean(lsd_frames), lsd_frames  # gennemsnit og alle frames

def compute_CC(original, measured):
    original_mean = np.mean(original)
    measured_mean = np.mean(measured)
    numerator = np.sum((original - original_mean) * (measured - measured_mean))
    denominator = np.sqrt(np.sum((original - original_mean)**2) * np.sum((measured - measured_mean)**2))
    cc = numerator / denominator
    return cc

def compute_cosine_similarity(original, measured):
    dot_product = np.dot(original, measured)
    norm_orig = np.linalg.norm(original)
    norm_meas = np.linalg.norm(measured)
    cosine_similarity = dot_product / (norm_orig * norm_meas)
    return cosine_similarity

def acoustic_contrast(rir,filter,wav_input,bright_zone_mics_index,dark_zone_mics_index):
    p_C=compute_pressure_with_input(rir,filter,wav_input)
    p_B=p_C[bright_zone_mics_index]
    p_D=p_C[dark_zone_mics_index]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(bright_zone_mics_index)
    M_D=len(dark_zone_mics_index)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return AC

def compute_STOI(original,measured): # pip install pystoi scipy
    """
    Calculate the Short-Time Objective Intelligibility (STOI) score between
    a reference and degraded audio file.

    Args:
        reference_file (str): Path to the reference WAV file.
        degraded_file (str): Path to the degraded WAV file.

    Returns:
        float: STOI score between 0.0 and 1.0.
    """
    # Load the reference and degraded audio
    fs_ref, ref = wavfile.read(original)
    fs_deg, deg = wavfile.read(measured)

    # Check sample rate consistency
    if fs_ref != fs_deg:
        raise ValueError("Sample rates of reference and degraded files must match.")

    # Convert stereo to mono if needed
    if ref.ndim > 1:
        ref = ref[:, 0]
    if deg.ndim > 1:
        deg = deg[:, 0]

    # Trim or pad both to same length
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    # Calculate STOI
    score = stoi(ref, deg, fs_ref, extended=False)
    return score
      


def performance_evaluation(
    test_features, test_filters, test_RIRs,
    original_wav_input, fs_wav,
    bright_zone_mics_index, dark_zone_mics_index
):
    """
    Simulates pressure fields for all test samples, saves degraded audio
    for bright and dark zones, and computes perceptual metrics.
    """
    save_dir="Performance Evaluation"
    os.makedirs(save_dir, exist_ok=True)

    # Save reference (original) audio for comparison
    ref_path = os.path.join(save_dir, "reference.wav")
    ref_np = original_wav_input.squeeze().cpu().numpy()
    ref_np /= np.max(np.abs(ref_np))
    wavfile.write(ref_path, fs_wav, (ref_np * 32767).astype(np.int16))

    results = []

    for i in range(len(test_features)):
        print(f"\n--- Evaluating sample {i+1}/{len(test_features)} ---")

        rir = test_RIRs[i]
        filter = test_filters[i]
        x_input = original_wav_input
        rir = rir.float().to(x_input.device)
        filter = filter.float().to(x_input.device)
        x_input = x_input.float().to(x_input.device)

        # --- 1. Compute acoustic pressure ---
        p = compute_pressure_with_input(rir, filter, x_input)
        
        # --- 2. Extract bright & dark zone pressures ---
        p_bright = p[bright_zone_mics_index[i]]
        p_dark   = p[dark_zone_mics_index[i]]

        p_bright_mean = torch.mean(p_bright, dim=0)
        p_dark_mean   = torch.mean(p_dark, dim=0)

        # --- 3. Convert to same type/shape as x_input ---
        p_bright_t = p_bright_mean.unsqueeze(0).to(dtype=x_input.dtype, device=x_input.device)
        p_dark_t   = p_dark_mean.unsqueeze(0).to(dtype=x_input.dtype, device=x_input.device)

        # Normalize (match input scaling)
        p_bright_t = p_bright_t / torch.max(torch.abs(p_bright_t))
        p_dark_t   = p_dark_t / torch.max(torch.abs(p_dark_t))

        # --- 4. Save degraded WAVs ---
        bright_path = os.path.join(save_dir, f"degraded_bright_{i}.wav")
        dark_path   = os.path.join(save_dir, f"degraded_dark_{i}.wav")

        # Convert to NumPy and ensure 2D for WAV: [N_samples, N_channels]
        p_bright_np = p_bright_t.cpu().numpy().reshape(-1, 1)
        p_dark_np   = p_dark_t.cpu().numpy().reshape(-1, 1)

        wavfile.write(bright_path, fs_wav, (p_bright_np * 32767).astype(np.int16))
        wavfile.write(dark_path,   fs_wav, (p_dark_np   * 32767).astype(np.int16))

        # --- 5. Compute metrics ---
        
        pesq_b = compute_pesq(ref_path, bright_path)
        pesq_d = compute_pesq(ref_path, dark_path)
        stoi_b = compute_STOI(ref_path, bright_path)
        stoi_d = compute_STOI(ref_path, dark_path)

        psnr_b = compute_psnr(ref_np, p_bright_np)
        psnr_d = compute_psnr(ref_np, p_dark_np)

        cc_b = compute_CC(ref_np, p_bright_np)
        cc_d = compute_CC(ref_np, p_dark_np)

        ac = acoustic_contrast(rir, filter, x_input, bright_zone_mics_index, dark_zone_mics_index)

        # --- 6. Store results ---
        results.append({
            "sample_idx": i,
            "PESQ_bright": pesq_b,
            "PESQ_dark": pesq_d,
            "STOI_bright": stoi_b,
            "STOI_dark": stoi_d,
            "PSNR_bright": psnr_b,
            "PSNR_dark": psnr_d,
            "CC_bright": cc_b,
            "CC_dark": cc_d,
            "Acoustic_Contrast": ac.item() if torch.is_tensor(ac) else ac
        })

        print(f"Results: PESQ_b={pesq_b:.2f}, PESQ_d={pesq_d:.2f}, STOI_b={stoi_b:.2f}, STOI_d={stoi_d:.2f}")
        print(f"         PSNR_b={psnr_b:.2f}, PSNR_d={psnr_d:.2f}, AC={ac:.2f}")

    return results

#print(RIRs.shape)
if __name__== "__main__":
    X_test=np.stack([X[0],X[1]],axis=0)
    n_srcs = 3
    filter_len = 1024

    # For the first two test points:
    filter_test = torch.stack([
        filters[0].reshape(n_srcs, filter_len),
        filters[1].reshape(n_srcs, filter_len)
    ], dim=0)
    test_RIRs = torch.stack([RIRs[0], RIRs[1]], dim=0).to(device)  # shape [2, 13, 3, 512]

    print(test_RIRs.shape)
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    x_input = x_input.to(device)
    # Original tensor
    bright_tensor = bright_zone_mics_index[0]  # the only element in the list
    dark_tensor   = dark_zone_mics_index[0]

    # Select first two test points (assuming first axis corresponds to data points)
    bright_zone_mics_index_test = [bright_tensor[0], bright_tensor[1]]
    dark_zone_mics_index_test   = [dark_tensor[0], dark_tensor[1]]


    print(performance_evaluation(X_test,filter_test,test_RIRs,x_input,fs_wav,bright_zone_mics_index_test,dark_zone_mics_index_test))
    

