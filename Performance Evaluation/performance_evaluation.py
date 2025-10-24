import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pesq
import torch
import scipy.io.wavfile as wavfile
from scipy.signal import convolve
import sys
import os

# Tilføj parent directory (P7-KODE) til Python-path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from MLP import FilterNet
from scipy.signal import stft
from VAST_filter_coefficients import setup_acoustic_scenario
from Dataset_generator_script import sources_mics, fs_target, room_dim#, #mic_directions, mic_positions_list, bright_zone_mics_index


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

def generate_cut_input(start, stop):

    fs_orig, wav = wavfile.read("relaxing-guitar-loop-v5-245859.wav")
    wav = np.mean(wav, axis=1)
    wav = wav[int(start * fs_orig):int(stop * fs_orig)].astype(np.float32)
    wav /= np.abs(wav).max()
    
    original_path = "input_sound_cut.wav"
    wavfile.write(original_path, fs_orig, (wav * 32767).astype(np.int16))
    print(f'Saved: {original_path}')
    
    return original_path

def generate_IR(): 
    '''
    Generate impulse responses for the given scenario and save to "test_ir.pt" to save time in future runs.
    '''
    
    IR = setup_acoustic_scenario(sources=sources_position_list, mic_positions_list=mic_positions_list, bright_zone_mics_index=bright_zone_mics_index, 
                            dark_zone_mics_index=dark_zone_mics_index, fs_target=fs_target, room_dim=room_dim, 
                            rt60=rt60, mic_directions=mic_directions, user_rotation=user_rotation)[0]
    torch.save(IR, "test_ir.pt")
    return IR

def generate_measured_path():
    '''
    Convolves original input with impulse responses and filter coefficients to get measured output.
    Saves the file "reproduced_sound.wav" to save time in future runs.
    '''

    X = np.concatenate([[rt60], [phone_tilt], [user_rotation], spatial_position])
    dumm = torch.tensor(X, dtype=torch.float32)

    
    input_size = 6 
    L, J = 3, 1024
    output_size = L * J

    model = FilterNet(input_size, output_size)
    model.load_state_dict(torch.load("filter_mlp_weights.pth"))
    model.eval()

    with torch.no_grad():
        Y = model(dumm).cpu().numpy().squeeze()
    q_matrix = Y.reshape(L, J)

    fs_orig, wav = wavfile.read("input_sound_cut.wav")
    IR = torch.load("test_ir.pt", weights_only=False)
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

    measured_path = "reproduced_sound.wav"
    wavfile.write(measured_path, fs_orig, (outputs * 32767).astype(np.int16))
    print(f"Saved: {measured_path}")
    return measured_path

def update_all():
    generate_cut_input(start, stop)
    generate_IR()
    generate_measured_path()



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

def compute_pesq(original, measured):
    pesq_score = pesq.pesq(16000, original, measured, 'nb')
    print(f"PESQ score: {pesq_score:.3f}")
    return pesq_score

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
    pesq_score = compute_pesq(x, y)
    CC_score = compute_CC(x, y)
    Cosine_sim = compute_cosine_similarity(x, y)


    print(f"PSNR: {psnr:.2f} dB")
    print(f"Log-Spectral Distance (LSD): {lsd_mean:.2f} dB")
    print(f"PESQ: {pesq_score:.2f}")
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


update_all()
original = "input_sound_cut.wav"
measured = "reproduced_sound.wav"
analyze_audio(original, measured)

