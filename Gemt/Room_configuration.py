import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import time, os



# -------------------------
# Parameters (tune these)
# -------------------------
wav_path = "Junk/Signe_sang.wav"
fs_target = 16000
J = 256                # filter length per loudspeaker
N = 2000               # number of time rows in U^m (samples)
grid_res = 50
z_plane = 1.5
reg_eps = 1e-6         # epsilon (regularization)
xi = 0.9               # weighting for dark leakage (0 = ignore dark penalty)
room_dim = [6.0, 3.0, 3.5]
absorption = 0.2
max_order = 10
# desired target amplitude scaling
target_amplitude = 1.0
# VAST parameters
V = 4
mu = 0.5
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

# ---------- room & geometry ----------
room_dim = [6.0, 3.0, 3.5]
absorption = 0.2
max_order = 10
room = pra.ShoeBox(room_dim, fs=fs_target, materials=pra.Material(absorption), max_order=max_order)

# speakers (L)
sources = [
    [0.5, 0.0, 1.5],
    [1.0, 0.0, 1.5],
    [1.5, 0.0, 1.5],
    [2.0, 0.0, 1.5],
    [2.5, 0.0, 1.5],
    [3.0, 0.0, 1.5],
    [3.5, 0.0, 1.5],
    [4.0, 0.0, 1.5],
    [4.5, 0.0, 1.5],
    [5.0, 0.0, 1.5],
    [5.5, 0.0, 1.5],
]
for s in sources:
    room.add_source(s)

# microphones (grid)
mic_positions_list = []
for x_m in np.linspace(0.5, 2.5, 6):
    for y_m in np.linspace(0.5, 2.5, 4):
        mic_positions_list.append([x_m, y_m, 1.5])
for x_m in np.linspace(3.5, 5.5, 6):
    for y_m in np.linspace(0.5, 2.5, 4):
        mic_positions_list.append([x_m, y_m, 1.5])

mic_positions = np.array(mic_positions_list).T
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
print("computing RIRs...")
t0 = time.perf_counter()
room.compute_rir()
print("RIRs computed in {:.2f} s".format(time.perf_counter() - t0))

# Convert room.rir to a more convenient list: IR[mic_idx][src_idx] -> numpy array
IR = room.rir
n_mics = mic_positions.shape[1]
n_srcs = len(sources)

# pad IRs to equal length K
K = max(len(IR[m][s]) for m in range(n_mics) for s in range(n_srcs))

# Pad all RIRs to the maximum length K
for m in range(n_mics):
    for s in range(n_srcs):
        if len(IR[m][s]) < K:
            IR[m][s] = np.pad(IR[m][s], (0, K - len(IR[m][s])), mode='constant')

# bright/dark indices (left half = bright)
mic_array = mic_positions.T
bright_zone_mics = [i for i,p in enumerate(mic_array) if p[0] < room_dim[0]/2]
dark_zone_mics   = [i for i,p in enumerate(mic_array) if p[0] >= room_dim[0]/2]
print("Bright mics:", len(bright_zone_mics), "Dark mics:", len(dark_zone_mics))

# -------------------------
# Helper: build U^{m,l} and U^m
# -------------------------
def build_U_ml_single_from_u(u_ml, N, J):
    """Toeplitz convolution matrix from precomputed u = x * h_{m,l}.
       u_ml length may be >= N; we use u_ml[:N] as first_col and zero-first-row.
    """
    first_col = u_ml[:N] if u_ml.shape[0] >= N else np.pad(u_ml, (0, N-u_ml.shape[0]))
    first_row = np.zeros(J)
    U_ml = toeplitz(first_col, first_row)[:N,:J]
    return U_ml

def build_Um_for_mic(m_idx, x, IR, N, J):
    """Build U^m by horizontally concatenating U^{m,l} for all l."""
    U_blocks = []
    for l in range(n_srcs):
        h_ml = np.array(IR[m_idx][l])
        u_ml = fftconvolve(x, h_ml)[:N]    # u_ml = x * h_ml
        if u_ml.shape[0] < N:
            u_ml = np.pad(u_ml, (0, N - u_ml.shape[0]))
        U_ml = build_U_ml_single_from_u(u_ml, N, J)
        U_blocks.append(U_ml)
    U_m = np.hstack(U_blocks)
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

# -------------------------
# Build stacked U_B, U_D, and desired d_B
# -------------------------

def build_U_and_dB():
    print("Building U matrices for all bright/dark mics (this may take time)...")
    tstart = time.perf_counter()
    U_B_list = []
    U_D_list = []
    # also build per-mic desired vector d_m for bright (we will set using a virtual source)
    for m in bright_zone_mics:
        U_m = build_Um_for_mic(m, x, IR, N, J)
        U_B_list.append(U_m)
    for m in dark_zone_mics:
        U_m = build_Um_for_mic(m, x, IR, N, J)
        U_D_list.append(U_m)
    U_B = np.vstack(U_B_list) if len(U_B_list)>0 else np.zeros((0, n_srcs*J))
    U_D = np.vstack(U_D_list) if len(U_D_list)>0 else np.zeros((0, n_srcs*J))
    print("Built U_B (shape {}) and U_D (shape {})".format(U_B.shape, U_D.shape))
    print("Time:", time.perf_counter()-tstart)
    return U_B, U_D

def build_R():
    U_B, U_D = build_U_and_dB()
    R_B = build_R_from_micset(bright_zone_mics, x, IR, N, J)
    R_D = build_R_from_micset(dark_zone_mics, x, IR, N, J)
    d_B = build_dB()
    r_d = U_B.T @ d_B      # (LJ,)
    # Regularize R_D slightly to ensure pos-definite for generalized eigenproblem
    R_D_reg = R_D + reg_eps * np.eye(R_D.shape[0])
    return R_B, R_D, R_D_reg, r_d



# -------------------------
# Build desired pressure d_B
# -------------------------

def build_dB():
    # We will construct d_B as the pressure that would be produced at the bright mics
    # by a virtual desired point-source placed at the center of the bright region.
    # This is a convenient and common choice: user may replace d_B with any desired waveform.
    bright_center = np.array([ (0.5+2.5)/2.0, (0.5+2.5)/2.0, z_plane ])  # center of bright region
    def virtual_rir_to_point(point, mic_pos, fs):
        # direct-path approximate RIR: single-sample delay at distance r, amplitude 1/r
        r = np.linalg.norm(point - mic_pos)
        delay = int(round(r * fs / 343.0))
        h = np.zeros(delay+1)
        h[delay] = 1.0 / (r + 1e-6)
        return h

    # build desired vector per bright mic (stacked)
    dB_blocks = []
    for m in bright_zone_mics:
        mic_pos = mic_array[m]
        hvir = virtual_rir_to_point(bright_center, mic_pos, fs_target)
        d_m = fftconvolve(x, hvir)[:N]
        if d_m.shape[0] < N:
            d_m = np.pad(d_m, (0, N - d_m.shape[0]))
        dB_blocks.append(target_amplitude * d_m)
    d_B = np.concatenate(dB_blocks) if len(dB_blocks)>0 else np.zeros((0,))
    return d_B