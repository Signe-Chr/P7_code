import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pesq
import torch
import scipy.io.wavfile as wavfile
from scipy.signal import convolve
from MLP import FilterNet
from scipy.signal import stft

""" Other metrics:
    MSE or RMSE
    Correlation, CC
    RSRQ (SNR lignende)
    SNR, PSNR eller PSIR (peak-snr eller sir)
    Log-Spectral Distance (LSD)
"""

wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
wav = np.mean(wav, axis=1)
wav = wav[5*44100:7*44100]
wav = wav.astype(np.float32)
wav /= np.max(np.abs(wav))  # normalize

input_size = 6   # adjust if you have 7 features
L, J = 3, 1024
output_size = L * J

model = FilterNet(input_size, output_size)
model.load_state_dict(torch.load("filter_mlp_weights.pth"))
model.eval()

dummy_input = np.array([2.2,    # Reverberation, float
                        0.78,      # Phone tilt, degrees
                        3.14,      # Orientation,  degrees
                        5, 10, 1.7])    # Spatial position, (x, y, z)       
dumm = torch.tensor(dummy_input, dtype=torch.float32)
print(dumm, dummy_input)

with torch.no_grad():
    Y = model(dumm).cpu().numpy().squeeze()
print(f"Predicted filter vector length: {len(Y)}")


# Reshape into L filters
L = 3        # number of loudspeakers
J = len(Y) // L  # filter order per loudspeaker
q_matrix = Y.reshape(L, J)
print(f"Filter matrix shape: {q_matrix.shape}")


# Convolve input with each filter
outputs = []
for l in range(L):
    y = convolve(wav, q_matrix[l], mode='full')
    outputs.append(y)

# Align to same length
min_len = min(len(y) for y in outputs)
outputs = np.stack([y[:min_len] for y in outputs], axis=1)

# Normalize to avoid clipping
outputs /= np.max(np.abs(outputs))


# Save reproduced audio
output_path = "reproduced_sound.wav"
wavfile.write(output_path, fs_wav, (outputs * 32767).astype(np.int16))
print(f"Saved: {output_path}")


# Evaluate PESQ (using mono reference)
# Combine channels (mono mixdown)
reproduced_mono = np.mean(outputs, axis=1)

# Ensure same length as reference
min_len = min(len(wav), len(reproduced_mono))
ref = wav[:min_len]
deg = reproduced_mono[:min_len]

# Compute PESQ (narrow-band)
pesq_score = pesq.pesq(16000, ref, deg, 'nb')
print(f"PESQ score: {pesq_score:.3f}")

#####################################################################################################

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

def analyze_audio(original_path, measured_path):
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

    print(f"PSNR: {psnr:.2f} dB")
    print(f"Log-Spectral Distance (LSD): {lsd_mean:.2f} dB")

    # Plot fejl over tid
    plt.figure(figsize=(10,4))
    plt.plot(lsd_frames)
    plt.title("Log-Spectral Distance per frame")
    plt.xlabel("Frame index")
    plt.ylabel("LSD (dB)")
    plt.grid(True)
    plt.show()

    return psnr, lsd_mean

# Then call:
psnr_val, lsd_val = analyze_audio(wav_path, output_path)
