import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import lfilter, fftconvolve
from scipy.linalg import toeplitz, eigh
import matplotlib.pyplot as plt
import time
import os

# -------------------------
# Parameters (tune these)
# -------------------------
wav_path = "Signe_sang.wav"   # change if needed
fs_target = 16000             # sampling rate used for room simulation
J = 256                       # filter length to design per loudspeaker
N = 2000                      # number of time samples (rows of U^m)
grid_res = 50                 # visualization grid resolution
z_plane = 1.5                 # height to visualize SPL
reg_eps = 1e-6                # regularization for R_D (for numerical stability)
# -------------------------

# -------------- Load signal --------------
if not os.path.exists(wav_path):
    raise FileNotFoundError(f"{wav_path} not found - adjust path.")
fs_wav, wav = wavfile.read(wav_path)
# Resample not implemented here; just truncate / convert
if fs_wav != fs_target:
    print(f"Warning: wav sample rate {fs_wav} != target {fs_target}. Results may be inconsistent.")
wav = np.array(wav, dtype=float)
# If stereo, take first channel
if wav.ndim > 1:
    wav = wav[:, 0]
# normalize the input signal to prevent clipping and set reference amplitude
wav = wav / (np.max(np.abs(wav)) + 1e-12)
# choose x (excitation) length
x = wav[:N].copy() if len(wav) >= N else np.pad(wav, (0, max(0, N-len(wav))), mode='constant')

# -------------- Build room and geometry --------------
room_dim = [6.0, 3.0, 3.5]
absorption = 0.2
max_order = 10

room = pra.ShoeBox(
    room_dim,
    fs=fs_target,
    materials=pra.Material(absorption),
    max_order=max_order,
)

# speaker positions (L sources along x=0 plane)
sources = [
    [0.5, 0.0, 1.5], [1.0, 0.0, 1.5], [1.5, 0.0, 1.5], [2.0, 0.0, 1.5], [2.5, 0.0, 1.5],
    [3.0, 0.0, 1.5], [3.5, 0.0, 1.5], [4.0, 0.0, 1.5], [4.5, 0.0, 1.5], [5.0, 0.0, 1.5],
    [5.5, 0.0, 1.5],
]
for s in sources:
    room.add_source(s)

# microphone grid (24 in bright zone, 24 in dark zone)
mic_positions_list = []
# left half (bright candidate)
for x_m in np.linspace(0.5, 2.5, 6):
    for y_m in np.linspace(0.5, 2.5, 4):
        mic_positions_list.append([x_m, y_m, 1.5])
# right half (dark candidate)
for x_m in np.linspace(3.5, 5.5, 6):
    for y_m in np.linspace(0.5, 2.5, 4):
        mic_positions_list.append([x_m, y_m, 1.5])

mic_positions = np.array(mic_positions_list).T
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

print("Room created. Computing RIRs (this may take a few seconds)...")
t0 = time.perf_counter()
room.compute_rir()
print("RIRs computed in {:.2f} s".format(time.perf_counter() - t0))

# Convert room.rir to a more convenient list: IR[mic_idx][src_idx] -> numpy array
IR = room.rir
n_mics = mic_positions.shape[1]
n_srcs = len(sources)

# confirm RIR lengths and pad them to same length K
K = max(len(IR[m][s]) for m in range(n_mics) for s in range(n_srcs))
print("Max RIR length K =", K)

# Pad all RIRs to the maximum length K
for m in range(n_mics):
    for s in range(n_srcs):
        if len(IR[m][s]) < K:
            IR[m][s] = np.pad(IR[m][s], (0, K - len(IR[m][s])), mode='constant')

# define bright & dark mic indices using actual x coordinate
mic_array = mic_positions.T
bright_zone_mics = [i for i,p in enumerate(mic_array) if p[0] < room_dim[0]/2]   # left half (x < 3.0)
dark_zone_mics   = [i for i,p in enumerate(mic_array) if p[0] >= room_dim[0]/2]  # right half (x >= 3.0)
print("Bright mics:", len(bright_zone_mics), "Dark mics:", len(dark_zone_mics))

# -------------------------
# Helper: build U^{m,l} blocks and U^m
# -------------------------
def build_U_ml_single(x, h_ml, N, J):
    """
    Build U^{m,l} (N x J) for a single mic m and single speaker l.
    This matrix represents the convolution of the excitation signal x with 
    the RIR h_ml (mic m, speaker l). It is the Toeplitz matrix formed from 
    the convolved signal, truncated to length N in time.
    """
    # u = x * h_ml (full conv), truncated to N samples
    u = np.convolve(x, h_ml)[:N]
    if u.shape[0] < N:
        u = np.pad(u, (0, N - u.shape[0]))
    
    # Create the Toeplitz matrix (first column is u, first row is zeros of length J)
    first_col = u
    first_row = np.zeros(J)
    U_ml = toeplitz(first_col, first_row)[:N, :J]
    return U_ml

def build_Um_for_mic(m_idx, x, IR, N, J):
    """
    Build U^m for microphone index m_idx by horizontally concatenating U^{m,l} for all l.
    Returns U_m (N x (L*J))
    """
    U_blocks = []
    for l in range(n_srcs):
        h_ml = IR[m_idx][l]
        U_ml = build_U_ml_single(x, h_ml, N, J)
        U_blocks.append(U_ml)
    U_m = np.hstack(U_blocks)   # (N, n_srcs*J)
    return U_m

# -------------------------
# Build R_B and R_D (Acoustic Pressure Covariance Matrices)
# -------------------------
def build_R_from_micset(mic_indices, x, IR, N, J):
    """
    Compute R = (1/|M|) * sum_{m in M} U^m.T @ U^m 
    where M is the set of microphones (bright or dark).
    R is the average covariance matrix over the microphones in the set.
    """
    LJ = n_srcs * J
    R = np.zeros((LJ, LJ), dtype=float)
    
    # Sum the contribution of each microphone's U^m matrix
    for m in mic_indices:
        U_m = build_Um_for_mic(m, x, IR, N, J)
        R += U_m.T @ U_m
        
    # Average across mics (consistent scaling)
    R /= max(1, len(mic_indices))
    # Additionally divide by N for a proper covariance approximation
    # R /= N 
    return R

print("Building R_B (bright) and R_D (dark). This may take some time...")

tstart = time.perf_counter()
R_B = build_R_from_micset(bright_zone_mics, x, IR, N, J)
R_D = build_R_from_micset(dark_zone_mics, x, IR, N, J)
print("Built R_B and R_D in {:.2f} s".format(time.perf_counter() - tstart))

print("R_B trace:", np.trace(R_B), "R_D trace:", np.trace(R_D))

# Regularize R_D slightly to ensure pos-definite for generalized eigenproblem
R_D_reg = R_D + reg_eps * np.eye(R_D.shape[0])

# Solve generalized eigenproblem R_B q = gamma R_D q
# The solution q is the eigenvector corresponding to the maximum eigenvalue (gamma)
print("Solving generalized eigenvalue problem (this may take a while for large LJ)...")
tstart = time.perf_counter()
eigvals, eigvecs = eigh(R_B, R_D_reg)   # returns ascending eigenvalues
t_eig = time.perf_counter() - tstart
print(f"Eigenproblem solved in {t_eig:.2f} s; number of eigvals = {len(eigvals)}")

# pick eigenvector with largest eigenvalue (highest acoustic contrast)
idx = np.argmax(eigvals)
gamma_max = eigvals[idx]
q_vec = eigvecs[:, idx]
print("Maximum Acoustic Contrast (gamma_max) =", gamma_max)

# q_vec may be complex if numerical issues; ensure real
q_vec = np.real(q_vec)

# scale / normalize q_vec to a sensible speaker amplitude level before plotting/simulation
# Normalization: max absolute coefficient of the concatenated filter vector is 1
q_vec = q_vec / (np.max(np.abs(q_vec)) + 1e-12)

# reshape q into per-source filters: (n_srcs, J)
q_matrix = q_vec.reshape(n_srcs, J)

print("q_matrix shape:", q_matrix.shape)

# -------------------------
# Compute resulting pressure field (uses direct path approximation for visualization)
# -------------------------
def pressure_field_from_q(q_matrix, IR, test_signal, sources, room_dim, fs_target, grid_res=50, z_plane=1.5):
    """
    Compute RMS pressure field (in pressure units, not dB) on a 2D grid at z=z_plane.
    NOTE: This uses the simple direct-path model for the visualization grid, NOT the full RIRs.
    """
    L = q_matrix.shape[0]
    x_grid = np.linspace(0, room_dim[0], grid_res)
    y_grid = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Gx, Gy = X.shape
    pressure_field = np.zeros_like(X, dtype=float)
    
    speed_of_sound = 343.0

    # Iterate over every point on the visualization grid
    for ix in range(Gx):
        for iy in range(Gy):
            point = np.array([X[ix,iy], Y[ix,iy], z_plane])
            p_sum = 0.0
            
            # The RMS value at the point is approximated as the sum of RMS contributions 
            # from each loudspeaker (a simple, non-coherent approximation)
            for l in range(L):
                # 1. Compute loudspeaker drive signal (convolved with the filter q_l)
                drive = fftconvolve(test_signal, q_matrix[l])[:len(test_signal)]
                
                # 2. Approximate acoustic path (direct path only)
                src_pos = np.array(sources[l])
                r = np.linalg.norm(point - src_pos)
                delay = int(round(r * fs_target / speed_of_sound))
                
                # Simple direct-path weight (1/r)
                if r > 1e-6:
                    h = np.zeros(max(1, delay + 1))
                    h[delay] = 1.0 / r 
                else:
                    h = np.array([1.0]) # Handles edge case where point is exactly at source
                    
                # The total pressure at the grid point is the convolution of 
                # the drive signal with the path approximation h.
                out = fftconvolve(drive, h)[:len(drive)]
                
                # Use the RMS value of the resulting pressure waveform as the measure
                p_sum += np.sqrt(np.mean(out**2) + 1e-12)
                
            pressure_field[ix,iy] = p_sum
            
    return X, Y, pressure_field

print("Computing pressure field (coarse grid for speed)...")
# Use a short segment of the signal for visualization to speed up convolution
test_signal = wav[:fs_target//4] if len(wav) >= fs_target//4 else wav
tstart = time.perf_counter()
# Note: we pass the sources list and fs_target to the function for direct path calculation
Xg, Yg, P = pressure_field_from_q(q_matrix, IR, test_signal, sources, room_dim, fs_target, grid_res=grid_res, z_plane=z_plane)
print("Pressure computed in {:.2f}s".format(time.perf_counter() - tstart))

# Compute averages for bright/dark masks
bright_mask = Xg < (room_dim[0]/2)
dark_mask = ~bright_mask
avg_bright = np.mean(P[bright_mask])
avg_dark = np.mean(P[dark_mask])
print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
print("Contrast (bright/dark) [dB] =", 20.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12)))

# plot SPL-like plot (convert to relative dB)
# Normalize to the maximum pressure on the grid for relative dB scale
P_db = 20.0 * np.log10(P / (np.max(P) + 1e-12) + 1e-12)
plt.figure(figsize=(8,6))
plt.imshow(P_db.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='inferno', aspect='auto')
plt.colorbar(label='Relative dB')
plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, edgecolors='k')
plt.title('Relative SPL (dB) at z={:.2f} m (Time-Domain MSIR)'.format(z_plane))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.show()

# Save q to file
out_q_path = "q_solution_td.npy"
np.save(out_q_path, q_matrix)
print("Saved q_matrix to", out_q_path)
