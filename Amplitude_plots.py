import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from Test_train_split import load_test_train_data, x_input

J=1024
#x_inp = [1] + [0]*(J+512-2)
x_inp=x_input.squeeze(0).numpy()
data_test, data_train, data_val = load_test_train_data()

q_filters=data_test[1][0]


RIRs_test=data_test[5][0]
bright_zone_IR=RIRs_test[12]
dark_zone_IR=RIRs_test[4]

# === Load filters ===
filters_random = torch.load("Saved Filters/random_selection_filters.pt")['selected_filters'][0].reshape(3, 1024)
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
    "Random": filters_random,
    "Baseline": filters_baseline,
    "Classification": filters_classification,
    "Regression": filters_regression,
    "Interpolation": filters_interpolation,
}

#wav = wav/max(abs(wav))
# === Compute and plot for each filter set ===
font = 10

bright_signal = resulting_amp_bright(q_filters)[:88200]
bright_original_signal = resulting_amp_original_bright(x_inp)[:88200]

norm = max(abs(bright_original_signal))

dark_original_signal = resulting_amp_original_dark(x_inp)[:88200]
dark_signal = resulting_amp_dark(q_filters)[:88200]

# normalize for fair comparison (optional)
bright_signal = bright_signal / norm
bright_original_signal = bright_original_signal / norm
dark_original_signal = dark_original_signal / norm
dark_signal = dark_signal / norm

# make time vector (samples to seconds)
t = np.arange(len(bright_signal)) / 16000

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
plt.savefig("ACC_fig_brightdark_subplots.pdf", dpi=500)
plt.show()

print("Subplot figure saved.")
