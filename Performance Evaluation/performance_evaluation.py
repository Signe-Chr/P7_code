import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import convolve, stft
from scipy.io import loadmat
from pesq import pesq
from scipy.io import wavfile
import torch.nn.functional as F
from pystoi import stoi
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
from MLP_classification import SoftFilterNet
from VAST_filter_coefficients import setup_acoustic_scenario
from Dataset_generator_script import sources_mics, fs_target, rooms #, #mic_directions, mic_positions_list, bright_zone_mics_index
from scipy.signal import resample

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
data_dir="Signes_data"
full_data=os.listdir(data_dir)
data_points = []
for data in full_data:
    if int(data.split("_")[1]) in ri:
        data_points.append(data)
data=CustomDataset(data_dir,data_points)
data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
Q=[batch for batch in data_loader][0]
X=Q[0]
filters=Q[1]
bright_zone_mics_index=Q[2]
dark_zone_mics_index=Q[3]
n_srcs=Q[4]
RIRs=Q[5]
"""
# -------------------------------------------------------------------------
# 1. Define scenario parameters
# -------------------------------------------------------------------------
# Choose segment of audio to evaluate
start = 1
stop = 6

#Input features for prediction
rt60 = 0.27                      # Reverberation, float: np.linspace(0.27, 0.7, 10)
phone_tilt = 1                   # Phone tilt, degrees in radians: 0.261, 0.785, 1.309
user_rotation = 1.57             # Orientation,  degrees in radians: 0, 1.57, 3.14, 4.71
spatial_position = np.array([5, 5, 1.7]).ravel()  # Spatial position (x, y, z): (5, 5 ,1.7) betyder i midten af rummet og i højde 1.7m               # flad ud til 1D

sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = sources_mics(R= 1 , Center = spatial_position , N_mics=12)

# -------------------------------------------------------------------------
# 2. Generate necessary files
# -------------------------------------------------------------------------


def generate_cut_input(start, stop, input):

    fs_orig, wav = wavfile.read(input)
    wav = np.mean(wav, axis=1)
    wav = wav[int(start * fs_orig):int(stop * fs_orig)].astype(np.float32)
    wav /= np.abs(wav).max()
    
    original_path = "Performance Evaluation/input_sound_cut.wav"
    wavfile.write(original_path, fs_orig, (wav * 32767).astype(np.int16))
    print(f'Saved: {original_path}')
    
    return original_path

def generate_IR(): 
    '''
    Generate impulse responses for the given scenario and save to "test_ir.pt" to save time in future runs.
    '''
    room_dim = rooms[1]  # Vælg et rum fra listen
    IR = setup_acoustic_scenario(sources=sources_position_list, mic_positions_list=mic_positions_list, bright_zone_mics_index=bright_zone_mics_index, 
                            dark_zone_mics_index=dark_zone_mics_index, fs_target=fs_target, room_dim=room_dim, 
                            rt60=rt60, mic_directions=mic_directions, user_rotation=user_rotation)[0]
    torch.save(IR, "Performance Evaluation/test_ir.pt")
    return IR

def generate_measured_path(original):
    '''
    Convolves original input with impulse responses and filter coefficients to get measured output.
    Saves the file "Performance Evaluation/reproduced_sound.wav" to save time in future runs.
    '''
    cut = original
    X = np.concatenate([[rt60], [phone_tilt], [user_rotation], spatial_position])
    dumm = torch.tensor(X, dtype=torch.float32)

    
    input_size = 6 
    L, J = 3, 1024
    output_size = L * J

    #model = FilterNet(input_size, output_size)
    #model.load_state_dict(torch.load("filter_mlp_weights.pth"))

    # Load any needed tensors or metadata
    filters_tensor = torch.load("filters_tensor.pt")  # if you saved it
    input_size = 6  # whatever your input size was
    num_filters = filters_tensor.shape[0]
    filter_dim = filters_tensor.shape[1]

    # Create model
    model = SoftFilterNet(input_size, num_filters, filter_dim, filters_tensor)
    model.load_state_dict(torch.load("mlp_weights.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Y = model(dumm).cpu().numpy().squeeze()
    q_matrix = Y.reshape(L, J)

    fs_orig, wav = wavfile.read("Performance Evaluation/input_sound_cut.wav")
    IR = torch.load("Performance Evaluation/test_ir.pt", weights_only=False)
    outputs = []
    for i in range(L):
        RIR = IR[bright_zone_mics_index[0]][i]  # Select RIR for bright zone mic and first source
        y = convolve(wav, RIR, mode='full')
        y2 = convolve(y, q_matrix[i], mode='full')
        outputs.append(y2)


    # Align to same length
    min_len = min(len(y) for y in outputs)
    outputs = np.stack([y[:min_len] for y in outputs], axis=1)

    # Normalize to avoid clipping
    outputs /= np.max(np.abs(outputs))

    measured_path = "Performance Evaluation/reproduced_sound.wav"
    wavfile.write(measured_path, fs_orig, (outputs * 32767).astype(np.int16))
    print(f"Saved: {measured_path}")
    return measured_path

def update_all():
    generate_cut_input(start, stop)
    generate_IR()
    generate_measured_path()
"""
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

def compute_pesq(original, measured,mode:str='wb'):
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
      
import os
import torch
import numpy as np
from scipy.io import wavfile

def performance_evaluation(
    test_features, test_filters, test_RIRs,
    original_wav_input, fs_wav,
    bright_zone_mics_index, dark_zone_mics_index,
    save_dir="Performance Evaluation"
):
    """
    Simulates pressure fields for all test samples, saves degraded audio
    for bright and dark zones, and computes perceptual metrics.
    """
    
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

print(RIRs.shape)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def analyze_audio(measured_path, original_path):
    fs_orig, x = wavfile.read(original_path)
    fs_meas, y = wavfile.read(measured_path)

    # Konverter til float mellem -1 og 1, hvis nødvendigt
    if x.dtype != np.float32:
        x = x.astype(np.float32) / np.max(np.abs(x))
    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.max(np.abs(y))
    
    # Convert to mono if multi-channel
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)  # Convert to mono by averaging channels
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono if needed

    # Sørg for samme længde
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    psnr = compute_psnr(x, y)
    lsd_mean, lsd_frames = compute_lsd(x, y, fs_orig)
    mos_score, pesq_score = compute_pesq(torch.asarray(x),torch.asarray(y))
    CC_score = compute_CC(x, y)
    Cosine_sim = compute_cosine_similarity(x, y)
    


    print(f"PSNR: {psnr:.2f} dB")
    print(f"Log-Spectral Distance (LSD): {lsd_mean:.2f} dB")
    print(f"PESQ: {pesq_score}")
    print(f"MOS: {mos_score}")
    print(f"Cross-Correlation (CC): {CC_score:.2f}")
    print(f"Cosine Similarity: {Cosine_sim:.2f}")

    # Plot fejl over tid
    plt.figure(figsize=(10,4))
    plt.plot(lsd_frames)
    plt.title("Log-Spectral Distance per frame")
    plt.xlabel("Frame index")
    plt.ylabel("LSD (dB)")
    plt.grid(True)
    plt.show()

    return psnr, lsd_mean


"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve

def apply_filter_to_audio(filter_path, input_wav_path, output_wav_path=None, normalize=True):
    '''
    Applies a FIR filter (from .txt file) to an input WAV file and saves the output.

    Parameters
    ----------
    filter_path : str
        Path to the .txt file containing the filter coefficients.
    input_wav_path : str
        Path to the input WAV file.
    output_wav_path : str, optional
        Path to save the filtered audio. If None, appends '_filtered.wav' to input filename.
    normalize : bool, default=True
        Whether to normalize output to prevent clipping.

    Returns
    -------
    output_wav_path : str
        The path of the saved filtered WAV file.
    '''

    # ---- 1. Load filter coefficients ----
    print(np.shape(np.loadtxt(filter_path, dtype=np.float32)))
    filt = np.loadtxt(filter_path, dtype=np.float32).reshape((3, 1024))[0]
    print(f"Loaded filter from '{filter_path}' with {len(filt)} coefficients.")

    # ---- 2. Load input audio ----
    fs, audio = wavfile.read(input_wav_path)
    print(f"Loaded '{input_wav_path}' with sample rate {fs} Hz.")

    # Convert stereo → mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert to float32 in [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))

    # ---- 3. Apply convolution ----
    #print(np.shape(audio),np.shape(filt))
    filtered_audio = convolve(audio, filt, mode='full')

    # ---- 4. Normalize ----
    if normalize:
        filtered_audio /= np.max(np.abs(filtered_audio))

    # ---- 5. Save output ----
    if output_wav_path is None:
        base = input_wav_path.rsplit('.', 1)[0]
        output_wav_path = f"{base}_filtered.wav"

    wavfile.write(output_wav_path, fs, (filtered_audio * 32767).astype(np.int16))
    print(f"Saved filtered audio to '{output_wav_path}'")

    return output_wav_path


#update_all()
#generate_measured_path()

import Dataset_generator_script as dgs

sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = dgs.sources_mics(dgs.dark_mic_radius, spatial_position, dgs.N_mics)


q, IR = design_vast_filter(sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                    dgs.wav, dgs.RT60s[0], mic_directions, dgs.user_rotations[0], dgs.fs_target, dgs.J, dgs.N, 
                    dgs.V, dgs.mu, [10, 10, 10], dgs.reg_eps, dgs.target_amplitude)

print(np.shape(q))

np.savetxt("predicted_filter_vast.txt", q, fmt="%.8f")
print(f"Filter saved to 'predicted_filter_vast.txt")
"""
"""
if __name__== "__main__":


    filter1 = "predicted_filter_top1.txt"
    filter2 = "predicted_filter_fnet_2.txt"
    filter3 = "predicted_filter_vast.txt"

    original = "relaxing-guitar-loop-v5-245859.wav"

    output_wav_path = "relaxing-guitar-loop-v5-245859_filterd.wav"
    #generate_cut_input(start, stop, original)

    #measured = apply_filter_to_audio(filter1, original, output_wav_path)

    #analyze_audio(original, measured)

    file = "Phone Zone Data/Proc_B4107_M0_P0R000_T0.mat"
    data = loadmat(file)
    print(data.keys())
"""