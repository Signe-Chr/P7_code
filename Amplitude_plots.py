import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

# === Load data ===
wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
wav = wav[5*44100:7*44100]

dict_ = np.load(
    r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Signes_data\VAST_0_0_0_0_0.npy",
    allow_pickle=True
).item()

bright_zone_IR = dict_['IR'][12]
dark_zone_IR = dict_['IR'][11]
q_filters = dict_['q_matrix']


# === Load filters ===
filters_random = torch.load("Saved Filters/random_selection_filters.pt")['selected_filters'][0].reshape(3, 1024)
filters_baseline = torch.load("Saved Filters/baseline_filters.pt")[0].reshape(3, 1024)
filters_classification = torch.load("Saved Filters/classification_filters.pt")[0].reshape(3, 1024)
filters_regression = torch.load("Saved Filters/regression_filters.pt")[0].reshape(3, 1024)
filters_interpolation = torch.load("Saved Filters/interpolation_filters.pt")[0].reshape(3, 1024)

# === Helper functions ===
def resulting_amp_bright(q):
    y_tot = 0
    for i in range(len(q)):
        y_temp = np.convolve(q[i], bright_zone_IR[i])
        y = np.convolve(y_temp, wav)
        y_tot += y
    return y_tot

def resulting_amp_dark(q):
    y_tot = 0
    for i in range(len(q)):
        y_temp = np.convolve(q[i], dark_zone_IR[i])
        y = np.convolve(y_temp, wav)
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

wav = wav/max(abs(wav))
# === Compute and plot for each filter set ===
for name, filt in filter_sets.items():
    bright_signal = resulting_amp_bright(q_filters)[:88200]
    dark_signal = resulting_amp_dark(q_filters)[:88200]
    
    # normalize for fair comparison (optional)
    bright_signal = bright_signal / max(abs(wav))
    dark_signal = dark_signal / max(abs(wav))
    
    # make time vector (samples to seconds)
    t = np.arange(len(bright_signal)) / fs_wav
    
    plt.figure(figsize=(10,6))
    plt.rcParams.update({"font.size": 17})
    plt.plot(t, wav, label='Original Signal', color='limegreen', alpha=0.2, linewidth=1.2)
    plt.plot(t, bright_signal, label='BZ Signal', color='orange', alpha=0.35,  linewidth=1.2)
    plt.plot(t, dark_signal, label='DZ Signal', color='blue', alpha=0.5, linewidth=1.2)

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Amplitude")
    #plt.title(f"Bright vs Dark Zone Signal — {name} Filters")
    #plt.legend(loc = "upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Vast_fig.pdf", dpi = 500)
    print(f"figure vast saved")
    plt.show()
