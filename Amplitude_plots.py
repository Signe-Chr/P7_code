import os, sys, torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from Test_train_split import load_test_train_data, x_input

J=1024
#x_inp = [1] + [0]*(J+512-2)
x_inp=x_input.squeeze(0).numpy()
data_test, data_train, data_val = load_test_train_data()

q_filters=data_test[1][2].reshape(3,1024)


RIRs_test=data_test[5][2]
bright_zone_IR=RIRs_test[12]
dark_zone_IR=RIRs_test[4]

# === Load filters ===
#filters_random = torch.load("Saved Filters/random_selection_filters.pt")['selected_filters'][0].reshape(3, 1024)
filters_baseline = torch.load("Saved Filters/baseline_filters.pt")[0].reshape(3, 1024)
filters_classification = torch.load("Saved Filters/classification_filters.pt")[0].reshape(3, 1024)
filters_regression = torch.load("Saved Filters/regression_filters.pt")[0].reshape(3, 1024)
filters_interpolation = torch.load("Saved Filters/interpolation_filters.pt")[0].reshape(3, 1024)

# === Helper functions ===
def resulting_amp_bright(q):
    y_tot = 0
    for i in range(3):
        y_temp = np.convolve(q[i], bright_zone_IR[i])
        y = np.convolve(y_temp, x_inp)
        y_tot += y
    return y_tot

def resulting_amp_original_bright(x):
    y = 0
    for i in range(3):
        y_temp = np.convolve(x, bright_zone_IR[i])
        y += y_temp
    return y

def resulting_amp_original_dark(x):
    y = 0
    for i in range(3):
        y_temp = np.convolve(x, dark_zone_IR[i])
        y += y_temp
    return y

def resulting_amp_dark(q):
    y_tot = 0
    for i in range(3):
        y_temp = np.convolve(q[i], dark_zone_IR[i])
        y = np.convolve(y_temp, x_inp)
        y_tot += y
    return y_tot

# === Filter sets ===
filter_sets = {
    "Baseline": filters_baseline,
    "Classification": filters_classification,
    "Regression": filters_regression,
    "Interpolation": filters_interpolation,
}

#wav = wav/max(abs(wav))
# === Compute and plot for each filter set ===
font = 10
filters=q_filters
bright_signal = resulting_amp_bright(filters)
bright_original_signal = resulting_amp_original_bright(x_inp)

norm = max(abs(bright_original_signal))

dark_original_signal = resulting_amp_original_dark(x_inp)
dark_signal = resulting_amp_dark(filters)

# normalize for fair comparison (optional)
bright_signal = bright_signal / norm
bright_original_signal = bright_original_signal / norm
dark_original_signal = dark_original_signal / norm
dark_signal = dark_signal / norm
min_len = min(len(bright_signal), len(bright_original_signal),
              len(dark_signal), len(dark_original_signal))

bright_signal = bright_signal[:min_len]
bright_original_signal = bright_original_signal[:min_len]
dark_signal = dark_signal[:min_len]
dark_original_signal = dark_original_signal[:min_len]

# Recompute time vector
t = np.arange(min_len) / 16000
# make time vector (samples to seconds)
#t = np.arange(len(bright_signal)) / 16000

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
plt.rcParams.update({"font.size": 17})

# ------------------------
# Subplot 1: Bright Zone
# ------------------------
ax1.plot(t, bright_original_signal, label='Unfiltered',
         color='limegreen', alpha=0.25, linewidth=1.2)
ax1.plot(t, bright_signal, label='ACC',
         color='orange', alpha=0.55, linewidth=1.2)

ax1.set_ylabel("Normalised Amplitude", fontsize=20)
ax1.set_title("Bright Zone Signals", fontsize=25)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# ------------------------
# Subplot 2: Dark Zone
# ------------------------
ax2.plot(t, dark_original_signal, label='Unfiltered',
         color='limegreen', alpha=0.25, linewidth=1.2)
ax2.plot(t, dark_signal, label='ACC',
         color='blue', alpha=0.55, linewidth=1.2)

ax2.set_xlabel("Time [s]", fontsize=20)
ax2.set_ylabel("Normalised Amplitude", fontsize=20)
ax2.set_title("Dark Zone Signals", fontsize=25)

ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

# Optional: force identical limits (if you want)
# ax1.set_xlim(t.min(), t.max())
# ax1.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("Plots/ACC_fig_brightdark_subplots.pdf", dpi=500)
plt.show()

print("Subplot figure saved.")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram

fs = 16000
n_fft = 512
hop_length = 128

# Compute spectrograms
S_bright_orig, _, _ = specgram(bright_original_signal, NFFT=n_fft, Fs=fs, noverlap=hop_length)
S_bright_filt, _, _ = specgram(bright_signal, NFFT=n_fft, Fs=fs, noverlap=hop_length)
S_dark_orig, _, _ = specgram(dark_original_signal, NFFT=n_fft, Fs=fs, noverlap=hop_length)
S_dark_filt, _, _ = specgram(dark_signal, NFFT=n_fft, Fs=fs, noverlap=hop_length)

# Convert to dB
S_bright_orig_db = 10 * np.log10(S_bright_orig + 1e-10)
S_bright_filt_db = 10 * np.log10(S_bright_filt + 1e-10)
S_dark_orig_db = 10 * np.log10(S_dark_orig + 1e-10)
S_dark_filt_db = 10 * np.log10(S_dark_filt + 1e-10)

# Global min/max for consistent color scale
vmin = min(S_bright_orig_db.min(), S_bright_filt_db.min(),
           S_dark_orig_db.min(), S_dark_filt_db.min())
vmax = max(S_bright_orig_db.max(), S_bright_filt_db.max(),
           S_dark_orig_db.max(), S_dark_filt_db.max())

# ------------------------
# Create figure without constrained_layout
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bright zone unfiltered
im0 = axes[0,0].imshow(S_bright_orig_db, aspect='auto', origin='lower',
                       extent=[0, len(bright_original_signal)/fs, 0, fs/2],
                       cmap='viridis', vmin=vmin, vmax=vmax)
axes[0,0].set_title("Bright Zone - Unfiltered")
axes[0,0].set_ylabel("Frequency [Hz]")
axes[0,0].set_xlabel("Time [s]")

# Bright zone filtered
im1 = axes[0,1].imshow(S_bright_filt_db, aspect='auto', origin='lower',
                       extent=[0, len(bright_signal)/fs, 0, fs/2],
                       cmap='viridis', vmin=vmin, vmax=vmax)
axes[0,1].set_title("Bright Zone - Filtered (ACC)")
axes[0,1].set_xlabel("Time [s]")

# Dark zone unfiltered
im2 = axes[1,0].imshow(S_dark_orig_db, aspect='auto', origin='lower',
                       extent=[0, len(dark_original_signal)/fs, 0, fs/2],
                       cmap='viridis', vmin=vmin, vmax=vmax)
axes[1,0].set_title("Dark Zone - Unfiltered")
axes[1,0].set_ylabel("Frequency [Hz]")
axes[1,0].set_xlabel("Time [s]")

# Dark zone filtered
im3 = axes[1,1].imshow(S_dark_filt_db, aspect='auto', origin='lower',
                       extent=[0, len(dark_signal)/fs, 0, fs/2],
                       cmap='viridis', vmin=vmin, vmax=vmax)
axes[1,1].set_title("Dark Zone - Filtered (ACC)")
axes[1,1].set_xlabel("Time [s]")

# ------------------------
# Adjust layout and add colorbar to the right
# ------------------------
plt.subplots_adjust(right=0.88, wspace=0.3, hspace=0.4)  # increase hspace
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im3, cax=cbar_ax)
cbar.set_label('Power [dB]')

plt.savefig("Plots/ACC_fig_spectrogram.pdf", dpi=500)
plt.show()

