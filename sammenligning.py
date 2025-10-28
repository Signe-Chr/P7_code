import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, fftfreq

# ------------------------------------------------------------
# 1. Load model predictions
# ------------------------------------------------------------
f1 = np.loadtxt("predicted_filter_1.txt")
f2 = np.loadtxt("predicted_filter_fnet_2.txt")

print("=== Loaded Model Filters ===")
print(f"Filter 1 shape: {f1.shape}")
print(f"Filter 2 shape: {f2.shape}")

# ------------------------------------------------------------
# 2. Load VAST reference filter
# ------------------------------------------------------------
vast_archive = np.load("VAST_filter_archive_730.npy", allow_pickle=True).item()

# You can choose any key from the archive; here we pick one automatically
# or manually select if you know which case corresponds to your test input.
vast_key = list(vast_archive.keys())[0]
print(f"Using VAST archive key: {vast_key}")

q_matrix = vast_archive[vast_key]["q_matrix"]
f3 = np.ravel(q_matrix)

print(f"VAST filter shape: {f3.shape}")

# ------------------------------------------------------------
# 3. Normalize lengths
# ------------------------------------------------------------
min_len = min(len(f1), len(f2), len(f3))
f1, f2, f3 = f1[:min_len], f2[:min_len], f3[:min_len]

# ------------------------------------------------------------
# 4. Normalize amplitude for fair comparison
# ------------------------------------------------------------
def normalize(sig):
    return sig / np.max(np.abs(sig))

f1n, f2n, f3n = normalize(f1), normalize(f2), normalize(f3)

# ------------------------------------------------------------
# 5. Define comparison metrics
# ------------------------------------------------------------
def metrics(a, b, label_a, label_b):
    diff = a - b
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    corr = np.corrcoef(a, b)[0, 1]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n--- {label_a} vs {label_b} ---")
    print(f"MSE: {mse:.8e}")
    print(f"MAE: {mae:.8e}")
    print(f"Correlation: {corr:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    return diff

diff_12 = metrics(f1n, f2n, "Model 1", "Model 2")
diff_13 = metrics(f1n, f3n, "Model 1", "VAST")
diff_23 = metrics(f2n, f3n, "Model 2", "VAST")

# ------------------------------------------------------------
# 6. Visualization — time-domain comparison
# ------------------------------------------------------------
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(f1n, label="Model 1", alpha=0.8)
plt.plot(f2n, label="Model 2", alpha=0.8)
plt.plot(f3n, label="VAST Reference", alpha=0.8)
plt.legend()
plt.title("Filter Coefficients (Normalized)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(diff_13, color="red", label="Model 1 - VAST")
plt.plot(diff_23, color="blue", alpha=0.5, label="Model 2 - VAST")
plt.legend()
plt.title("Differences w.r.t. VAST Reference")
plt.grid(True)

plt.subplot(3, 1, 3)
corr_vast = correlate(f3n - np.mean(f3n), f1n - np.mean(f1n), mode="full")
lags = np.arange(-len(f3n) + 1, len(f1n))
plt.plot(lags, corr_vast)
plt.title("Cross-correlation (VAST vs Model 1)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Frequency-domain comparison (FFT)
# ------------------------------------------------------------
def plot_fft_comparison(f1, f2, f3, fs=16000):
    plt.figure(figsize=(14, 8))
    N = len(f1)
    freq = fftfreq(N, 1/fs)[:N//2]
    F1, F2, F3 = np.abs(fft(f1)[:N//2]), np.abs(fft(f2)[:N//2]), np.abs(fft(f3)[:N//2])

    plt.plot(freq, 20*np.log10(F1/np.max(F1)), label="Model 1")
    plt.plot(freq, 20*np.log10(F2/np.max(F2)), label="Model 2")
    plt.plot(freq, 20*np.log10(F3/np.max(F3)), label="VAST Reference")
    plt.title("Magnitude Spectrum (dB)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_fft_comparison(f1n, f2n, f3n)
