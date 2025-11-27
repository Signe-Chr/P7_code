import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import fftconvolve  # faster convolution

# === Paths ===
parent_dir = r"c:\Users\Signe Christensen\Downloads\Aalborg universitet\Matematik teknologi\7.semester\Projekt"

wav_path = os.path.join(parent_dir, "P7", "relaxing-guitar-loop-v5-245859.wav")
npy_path = os.path.join(parent_dir, "P7", "Signes_data", "VAST_0_0_0_0_0.npy")

# === Load audio ===
fs_wav, wav = wavfile.read(wav_path)
wav = wav[5*44100:7*44100]  # take 2-second snippet
wav = wav.astype(np.float32)
#wav /= np.max(np.abs(wav))  # normalize input audio to [-1,1]

# === Load IRs and filters ===
dict_ = np.load(npy_path, allow_pickle=True).item()
bright_zone_IR = np.array(dict_['IR'][12])
dark_zone_IR = np.array(dict_['IR'][4])
q_filters = dict_['q_matrix']

filters_random = torch.load(os.path.join(parent_dir, "P7", "Saved Filters", "random_selection_filters.pt"))['selected_filters'][0].reshape(3, 1024)
filters_baseline = torch.load(os.path.join(parent_dir, "P7", "Saved Filters", "baseline_filters.pt"))[0].reshape(3, 1024)
filters_classification = torch.load(os.path.join(parent_dir, "P7", "Saved Filters", "classification_filters.pt"))[0].reshape(3, 1024)
filters_regression = torch.load(os.path.join(parent_dir, "P7", "Saved Filters", "regression_filters.pt"))[0].reshape(3, 1024)
filters_interpolation = torch.load(os.path.join(parent_dir, "P7", "Saved Filters", "interpolation_filters.pt"))[0].reshape(3, 1024)

filter_sets = {
    "Random": filters_random,
    "Baseline": filters_baseline,
    "Classification": filters_classification,
    "Regression": filters_regression,
    "Interpolation": filters_interpolation,
}

# === Helper functions ===
def rms(x):
    return np.sqrt(np.mean(x**2))

def normalize_rms(x, target_rms=0.04):
    """Normalize signal to a target RMS"""
    current_rms = rms(x)
    gain = target_rms / (current_rms + 1e-12)
    y = x * gain
    return y

def convolve_with_IRs(wav, IRs):
    """Convolve wav with 3-channel IRs and sum"""
    y = 0
    for i in range(3):
        y += fftconvolve(wav, IRs[i])
    return y

def apply_filters_and_IRs(wav, filters, IRs):
    y_tot = 0
    for i in range(3):
        filtered = fftconvolve(wav, filters[i])
        y_tot += fftconvolve(filtered, IRs[i])
    return y_tot


wav = normalize_rms(wav)
# === Propagation without filters ===
bright_original_signal = convolve_with_IRs(wav, bright_zone_IR)
dark_original_signal = convolve_with_IRs(wav, dark_zone_IR)

# === Apply precomputed filters (speaker filters) ===
bright_signal = apply_filters_and_IRs(wav, q_filters,bright_zone_IR)
dark_signal = apply_filters_and_IRs(wav, q_filters,dark_zone_IR)

# === Trim after propagation ===
bright_signal = bright_signal[:88200]
bright_original_signal = bright_original_signal[:88200]
dark_signal = dark_signal[:88200]
dark_original_signal = dark_original_signal[:88200]
print("RMS of bright_original_signal:", rms(bright_original_signal))
print("RMS of dark_original_signal:", rms(dark_original_signal))
print("RMS of bright_signal:", rms(bright_signal))
print("RMS of dark_signal:", rms(dark_signal))
print("Max abs of q_filters:", np.max(np.abs(q_filters)))
# === Time vector ===
t = np.arange(len(bright_signal)) / fs_wav


# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
plt.rcParams.update({"font.size": 17})

# Bright zone
ax1.plot(t, bright_original_signal, label='Unfiltered', color='limegreen', alpha=0.25, linewidth=1.2)
ax1.plot(t, bright_signal, label='VAST', color='orange', alpha=0.55, linewidth=1.2)
ax1.set_ylabel("Normalized Amplitude", fontsize=20)
ax1.set_title("Bright Zone Signals", fontsize=25)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

# Dark zone
ax2.plot(t, dark_original_signal, label='Unfiltered', color='limegreen', alpha=0.25, linewidth=1.2)
ax2.plot(t, dark_signal, label='VAST', color='blue', alpha=0.55, linewidth=1.2)
ax2.set_xlabel("Time [s]", fontsize=20)
ax2.set_ylabel("Normalized Amplitude", fontsize=20)
ax2.set_title("Dark Zone Signals", fontsize=25)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

plt.tight_layout()
plt.savefig(os.path.join(parent_dir, "Vast_fig_brightdark_subplots_rms.pdf"), dpi=500)
plt.show()

print("Subplot figure saved with RMS normalization.")
