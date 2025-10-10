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
wav_path = "Signe_sang.wav"
fs_target = 16000
J = 256                # filter length per loudspeaker
N = 2000               # number of time rows in U^m (samples)
grid_res = 50
z_plane = 1.5
reg_eps = 1e-6         # epsilon (regularization)
xi = 0.9               # weighting for dark leakage (0 = ignore dark penalty)
# desired target amplitude scaling
target_amplitude = 1.0
# -------------------------

# ---------- load signal ----------
if not os.path.exists(wav_path):
    raise FileNotFoundError(f"{wav_path} not found")
fs_wav, wav = wavfile.read(wav_path)
if fs_wav != fs_target:
    print(f"Warning: wav sample rate {fs_wav} != target {fs_target}. Results may vary.")
wav = np.array(wav, dtype=float)
if wav.ndim > 1:
    wav = wav[:,0]
wav = wav / (np.max(np.abs(wav)) + 1e-12)
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
n_srcs = len(sources)

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

IR = room.rir
n_mics = mic_positions.shape[1]
n_srcs = len(sources)

# pad IRs to equal length K
K = max(len(IR[m][s]) for m in range(n_mics) for s in range(n_srcs))
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
# helper: build U^{m,l} and U^m
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
# Build stacked U_B, U_D, and desired d_B
# -------------------------
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

# -------------------------
# Build desired pressure d_B
# -------------------------
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

# -------------------------
# Compute PM solution
# -------------------------
print("Forming R_B, R_D and r_B...")
tstart = time.perf_counter()
R_B = U_B.T @ U_B      # (LJ x LJ)
R_D = U_D.T @ U_D      # (LJ x LJ)
r_B = U_B.T @ d_B      # (LJ,)
# normal eqn matrix
A = R_B + xi * R_D + reg_eps * np.eye(R_B.shape[0])
b = r_B  # note d_D=0 -> U_D.T d_D = 0
print("Solving linear system A q = b ...")
q_vec = np.linalg.solve(A, b)   # PM solution
t_eig = time.perf_counter() - tstart
print("Solved in {:.2f}s".format(t_eig))

# reshape to per-source filters
q_vec = np.real(q_vec)  # numerical safety
q_vec = q_vec / (np.max(np.abs(q_vec)) + 1e-12)   # normalize for safe playback / plotting
q_matrix = q_vec.reshape(n_srcs, J)
print("q_matrix shape:", q_matrix.shape)

# -------------------------
# Evaluate (same pressure_field routine as before)
# -------------------------
def pressure_field_from_q(q_matrix, IR, test_signal, room_dim, grid_res=50, z_plane=1.5):
    L = q_matrix.shape[0]
    x_grid = np.linspace(0, room_dim[0], grid_res)
    y_grid = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Gx, Gy = X.shape
    pressure_field = np.zeros_like(X, dtype=float)
    for ix in range(Gx):
        for iy in range(Gy):
            point = np.array([X[ix,iy], Y[ix,iy], z_plane])
            p_sum = 0.0
            for l in range(L):
                drive = fftconvolve(test_signal, q_matrix[l])[:len(test_signal)]
                src_pos = np.array(sources[l])
                r = np.linalg.norm(point - src_pos)
                delay = int(round(r * fs_target / 343.0))
                h = np.zeros(max(1, delay+1))
                h[delay] = 1.0/(r + 1e-6)
                out = fftconvolve(drive, h)[:len(drive)]
                p_sum += np.sqrt(np.mean(out**2) + 1e-12)
            pressure_field[ix,iy] = p_sum
    return X, Y, pressure_field

print("Computing pressure field for visualization...")
test_signal = wav[:fs_target//4] if len(wav) >= fs_target//4 else wav
tstart = time.perf_counter()
Xg, Yg, P = pressure_field_from_q(q_matrix, IR, test_signal, room_dim, grid_res=grid_res, z_plane=z_plane)
print("Computed in {:.2f}s".format(time.perf_counter() - tstart))

bright_mask = Xg < (room_dim[0]/2)
dark_mask = ~bright_mask
avg_bright = np.mean(P[bright_mask])
avg_dark = np.mean(P[dark_mask])
print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
print("Contrast (bright/dark) [dB] =", 20.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12)))

P_db = 20.0 * np.log10(P / (np.max(P) + 1e-12) + 1e-12)
plt.figure(figsize=(8,6))
plt.imshow(P_db.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='inferno', aspect='auto')
plt.colorbar(label='Relative dB')
plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, edgecolors='k')
plt.title('Relative SPL (dB) at z={:.2f} m (PM)'.format(z_plane))
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.tight_layout()
plt.show()

# Save q
np.save("q_PM.npy", q_matrix)
print("Saved q_PM.npy")
