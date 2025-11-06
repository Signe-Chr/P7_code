import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.signal import convolve, stft, resample
from scipy.io import loadmat, wavfile
from pesq import pesq
from pystoi import stoi
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
from MLP_classification import SoftFilterNet




def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs and filters with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        filter_q: [n_srcs, filter_len]
        reference: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = reference.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_input_samples    # We would like the output to match input as we are converting a sound signal
    
    # Zero pad reference for convolution
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
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
            
            # Convolve h_combined with input signal x (reference) using FFT
            
            # Pad h_combined to ensure final output length matches 'output_len'
            h_combined_padded = F.pad(h_combined, (0, output_len - h_combined.shape[0]), 'constant', 0)
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p

# -------------------------------------------------------------------------
# 3. Performance Evaluation Functions
# -------------------------------------------------------------------------

def compute_pesq(reference, measured, mode:str='wb'):
    """
    Calculate PESQ (MOS-LQO) score between a reference and measured audio file.

    Args:
        reference_file (str): Path to the reference WAV file.
        degraded_file (str): Path to the degraded WAV file.
        mode (str): 'nb' for narrowband (8 kHz) or 'wb' for wideband (16 kHz). Default is 'wb'.

    Returns:
        float: PESQ score (MOS-LQO) ranging approximately from 1.0 to 4.5.
    """
    fs_ref, wav_ref = wavfile.read(reference)
    fs_deg, wav_deg = wavfile.read(measured)

    # Check that sample rates match
    if fs_ref != fs_deg:
        raise ValueError("Sample rates of reference and degraded files do not match.")

    # Make sure the signals are mono (1D arrays)
    if wav_ref.ndim > 1:
        wav_ref = wav_ref[:, 0]
    if wav_deg.ndim > 1:
        wav_deg = wav_deg[:, 0]

    # Trim or pad signals to the same length
    min_len = min(len(wav_ref), len(wav_deg))
    wav_ref = wav_ref[:min_len]
    wav_deg = wav_deg[:min_len]
    wav_ref = resample_to_16k_np(reference)
    wav_deg = resample_to_16k_np(measured)
    score = pesq(16000, wav_ref, wav_deg, mode)
    # Calculate PESQ score
    #mos = pesq.mos(ref, deg)
    return score

def compute_psnr(reference, measured):
    mse = np.mean((reference - measured)**2)
    if mse == 0:
        return np.inf
    max_val = np.max(np.abs(reference))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def compute_lsd(reference, measured, fs, n_fft=1024, hop_length=512):
    # Short-Time Fourier Transform
    f1, t1, Z_orig = stft(reference, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    f2, t2, Z_meas = stft(measured, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Power spectrum in dB
    S_orig = 20 * np.log10(np.abs(Z_orig) + 1e-10)
    S_meas = 20 * np.log10(np.abs(Z_meas) + 1e-10)
    
    # LSD per frame
    lsd_frames = np.sqrt(np.mean((S_orig - S_meas)**2, axis=0))
    return np.mean(lsd_frames), lsd_frames  # gennemsnit og alle frames

def compute_CC(reference, measured):
    min_len = min(len(reference), len(measured))
    reference = reference[:min_len]
    measured = measured[:min_len]
    reference_mean = np.mean(reference)
    measured_mean = np.mean(measured)
    
    numerator = np.sum((reference - reference_mean) * (measured - measured_mean))
    denominator = np.sqrt(np.sum((reference - reference_mean)**2) * np.sum((measured - measured_mean)**2))
    
    cc = numerator / denominator
    return cc

def compute_cosine_similarity(reference, measured):
    dot_product = np.dot(reference, measured)
    norm_orig = np.linalg.norm(reference)
    norm_meas = np.linalg.norm(measured)
    cosine_similarity = dot_product / (norm_orig * norm_meas)
    return cosine_similarity

def acoustic_contrast(rir,filter, wav_input, bright_zone_mics_index,dark_zone_mics_index):
    p_C=compute_pressure_with_input(rir, filter, wav_input)
    p_B=p_C[bright_zone_mics_index]
    p_D=p_C[dark_zone_mics_index]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(bright_zone_mics_index)
    M_D=len(dark_zone_mics_index)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return AC

def compute_STOI(reference,measured):
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
    fs_ref, ref = wavfile.read(reference)
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

def load_data():
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
    return Q, device

def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs and filters with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        filter_q: [n_srcs, filter_len]
        reference: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = reference.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
    # Zero pad reference for convolution
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
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
            
            # Convolve h_combined with input signal x (reference) using FFT
            
            # Pad h_combined to ensure final output length matches 'output_len'
            h_combined_padded = F.pad(h_combined, (0, output_len - h_combined.shape[0]), 'constant', 0)
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p


def compute_pressure_with_input2(rir: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs directly with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        reference: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    n_input_samples = reference.shape[-1]
    
    # Output length = n_rir_samples + n_input_samples - 1
    output_len = n_rir_samples + n_input_samples - 1
    
    # Zero pad input
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    # FFT length (power of 2 for efficiency)
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))
    X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)

    # Loop through microphones and sources
    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            h = rir[m, s, :]  # [n_rir_samples]
            h_padded = F.pad(h, (0, output_len - n_rir_samples), 'constant', 0)
            H_fft = torch.fft.rfft(h_padded, n=n_fft)
            
            # Convolution via multiplication in frequency domain
            P_fft = H_fft * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len]
            p_m += p_m_s

        p[m, :] = p_m

    return p

def generate_measured_path(filters, unfiltered):
    '''
    Convolves the received sound with the filters.
    '''
    print(filters)
    print(unfiltered)
    filtered = []
    for i in range(len(filters)):  # For each source
        y = convolve(filters[i], unfiltered[i], mode='full')
        filtered.append(y)

    # Align to same length
    min_len = min(len(y) for y in filtered)
    filtered = np.stack([y[:min_len] for y in filtered], axis=1)

    # Normalize to avoid clipping
    filtered /= np.max(np.abs(filtered))
    return filtered

def performance_evaluation(
    test_features, filters, RIRs,
    reference, fs_wav,
    bright_zone_mics_index, dark_zone_mics_index, unfiltered_pressure, save=False
):
    """
    Simulates pressure fields for all test samples, saves degraded audio
    for bright and dark zones, and computes perceptual metrics.
    """
    save_dir="Performance Evaluation"
    os.makedirs(save_dir, exist_ok=True)

    # Save reference (reference) audio for comparison
    ref_path = os.path.join(save_dir, "reference.wav")


    ref_np = reference.squeeze().cpu().numpy()
    ref_np /= np.max(np.abs(ref_np))
    ref_np = np.asarray(ref_np, dtype=np.float32).ravel()  # <-- gør 1D
    reference = ref_np[:]

    wavfile.write(ref_path, fs_wav, (ref_np * 32767).astype(np.int16))

    results = []

    for i in range(len(test_features)):
        print(f"\n--- Evaluating sample {i+1}/{len(test_features)} ---")

        rir = RIRs[i]
        filter = filters[i]
        rir = rir.float().to(reference.device)
        filter = filter.float().to(reference.device)
        reference = reference.float().to(reference.device)

        # --- 1. Compute acoustic pressure ---
        filtered = compute_pressure_with_input(rir, filter, reference)
        
        # --- 2. Extract bright & dark zone pressures ---
        p_bright = filtered[bright_zone_mics_index[i]]
        p_dark   = filtered[dark_zone_mics_index[i]]
        p_dark_mean = torch.mean(p_dark, dim=0)

        # --- 3. Convert to same type/shape as reference ---
        p_bright_t = p_bright.to(dtype=reference.dtype, device=reference.device)
        p_dark_t   = p_dark_mean.unsqueeze(0).to(dtype=reference.dtype, device=reference.device)

        # Normalize (match input scaling)
        p_bright_t = p_bright_t / torch.max(torch.abs(p_bright_t))
        p_dark_t   = p_dark_t / torch.max(torch.abs(p_dark_t))

        
        bright_path = os.path.join(save_dir, f"degraded_bright_{i}.wav")
        dark_path   = os.path.join(save_dir, f"degraded_dark_{i}.wav")

        # Convert to NumPy and ensure 2D for WAV: [N_samples, N_channels]
        p_bright_np = p_bright_t.cpu().numpy().reshape(-1, 1)
        p_dark_np   = p_dark_t.cpu().numpy().reshape(-1, 1)
        p_bright_np = np.asarray(p_bright_np, dtype=np.float32).ravel() # <-- gør 1D
        p_dark_np   = np.asarray(p_dark_np, dtype=np.float32).ravel() # <-- gør 1D

        # --- 4. Save degraded WAVs ---
        if save == True:
            wavfile.write(bright_path, fs_wav, (p_bright_np * 32767).astype(np.int16))
            wavfile.write(dark_path,   fs_wav, (p_dark_np   * 32767).astype(np.int16))

        # --- 5. Compute metrics ---
        pesq_b = compute_pesq(ref_path, bright_path)
        pesq_d = compute_pesq(ref_path, dark_path)

        stoi_b = compute_STOI(ref_path, bright_path)
        stoi_d = compute_STOI(ref_path, dark_path)

        psnr_b = compute_psnr(ref_path, p_bright_np)
        psnr_d = compute_psnr(ref_path, p_dark_np)

        cc_b = compute_CC(ref_path, p_bright_np)
        cc_d = compute_CC(ref_path, p_dark_np)

        ac = acoustic_contrast(rir, filter, reference, bright_zone_mics_index, dark_zone_mics_index)

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
        print(f"         CC_b={cc_b:.2f},     CC_d={cc_d:.2f},     PSNR_b={psnr_b:.2f}, PSNR_d={psnr_d:.2f}")
        print(f"         AC={ac:.2f}")

    return results





#print(RIRs.shape)
if __name__== "__main__":
    X, filters, bright_zone_mics_index, dark_zone_mics_index, n_srcs, RIRs= load_data()[0]
    device = load_data()[1]
    bright_zone_mics_index = np.array(bright_zone_mics_index).T
    dark_zone_mics_index = np.array(dark_zone_mics_index).T

    X_test=np.stack([X[0],X[1]],axis=0)
    n_srcs = n_srcs[0]
    filter_len = len(filters[0])//n_srcs
    # For the first two test points:
    filter_test = torch.stack([
        filters[0].reshape(n_srcs, filter_len),
        filters[1].reshape(n_srcs, filter_len)
    ], dim=0)
    test_RIRs = torch.stack([RIRs[0], RIRs[1]], dim=0).to(device)  # shape [2, 13, 3, 512]

    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    reference = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    reference = reference.to(device)
    # reference tensor
    bright_tensor = bright_zone_mics_index[0]  # the only element in the list
    dark_tensor   = dark_zone_mics_index[0]

    # Select first two test points (assuming first axis corresponds to data points)
    bright_zone_mics_index_test = [bright_zone_mics_index[0], bright_zone_mics_index[1]]
    dark_zone_mics_index_test   = [dark_zone_mics_index[0], dark_zone_mics_index[1]]

    performance_evaluation(X_test, filter_test, test_RIRs, reference, fs_wav, bright_zone_mics_index_test, dark_zone_mics_index_test)

