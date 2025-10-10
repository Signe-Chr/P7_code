import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import lfilter, fftconvolve
from scipy.linalg import toeplitz, eigh
import matplotlib.pyplot as plt
import time
from scipy.linalg import eig
import os

# -------------------------
# Parameters 
# -------------------------
wav_path = "Signe_sang.wav"   # change if needed
fs_target = 16000             # sampling rate used for room simulation
J = 256                       # filter length to design per loudspeaker
N = 2000                      # number of time samples (rows of U^m)
grid_res = 50                 # visualization grid resolution
z_plane = 1.5                 # height to visualize SPL
reg_eps = 1e-6                # regularization for R_D (for numerical stability)
V = 4
mu = 0.5
room_dim = [6.0, 3.0, 3.5]
absorption = 0.2
max_order = 10
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


def setup_acoustic_scenario(sources, 
                            mic_positions_list, 
                            bright_zone_mics_index, 
                            dark_zone_mics_index, 
                            fs_target=16000):
    """
    Sets up a pyroomacoustics simulation environment (ShoeBox) with 
    custom source, microphone positions, and per-microphone directivity.
    
    Microphones specified in cardioid_mics_indices receive a cardioid pattern
    with a custom orientation; all others are omnidirectional.

    Args:
        sources (list of lists): Coordinates of the sound sources.
        mic_positions_list (list of lists): Coordinates of all microphones.
        bright_zone_mics_index (list): Indices (0-based) of mics in the bright zone.
        dark_zone_mics_index (list): Indices (0-based) of mics in the dark zone.
        fs_target (int): Target sample rate for the simulation (Hz).
        room_dim (list): [L, W, H] dimensions of the room.
        absorption (float): Wall absorption coefficient.
        max_order (int): Maximum reflections to compute.
        cardioid_mics_indices (list, optional): Indices of microphones that should
                                                have a cardioid pattern. Defaults to None (all omni).
        cardioid_orientations (list of np.ndarray, optional): A list of 
                                                [yaw, pitch, roll] orientations (in radians) 
                                                for each cardioid microphone. Must match 
                                                the length and order of cardioid_mics_indices.

    Returns:
        tuple: (room, IR, mic_positions, sources_list, mic_directivities, M_b, M_d, bright_zone_mics, dark_zone_mics)
            room (pra.ShoeBox): The simulated room object.
            IR (list): Room Impulse Responses in the format IR[mic_idx][src_idx].
            mic_positions (np.ndarray): 3xN array of microphone coordinates.
            sources_list (list): List of source coordinates (input).
            mic_directivities (list): List of Directivity objects or None (for omni).
            M_b (int): Number of bright zone microphones.
            M_d (int): Number of dark zone microphones.
            bright_zone_mics (list): Coordinates of bright zone mics.
            dark_zone_mics (list): Coordinates of dark zone mics.
    """
    # --- Check Inputs and Define Zone Counts ---
    sources_list = sources 

    # Define M_b and M_d based on the provided indices
    M_b, M_d = len(bright_zone_mics_index), len(dark_zone_mics_index)
    n_mics_total = len(mic_positions_list)

    if n_mics_total != (M_b + M_d):
        print(f"Warning: Total mics in list ({n_mics_total}) does not equal M_b + M_d ({M_b + M_d}). This may indicate a mapping error.")

    # --- 2. Define Room ---
    room = pra.ShoeBox(
        room_dim,
        fs=fs_target,
        materials=pra.Material(absorption),
        max_order=max_order,
    )

    # --- 3. Add Sources ---
    for s in sources_list:
        # Add sources to the room (default source directivity is isotropic/omnidirectional)
        room.add_source(s)

    # --- 4. Define and Add Microphone Grid ---
    # Convert list of coordinates to the required 3xN NumPy array format
    mic_positions = np.array(mic_positions_list).T

    # Create MicrophoneArray, passing the list of directivity objects
    mic_array = pra.MicrophoneArray(
        mic_positions,
        room.fs)
    room.add_microphone_array(mic_array)

    # --- 5. Compute RIRs ---
    print(f"Computing RIRs for {mic_positions.shape[1]} mics (Bright: {M_b}, Dark: {M_d}) and {len(sources_list)} sources...")
    room.compute_rir()

    # --- 6. Prepare Return Variables ---
    # RIRs are stored in room.rir: room.rir[mic_index][source_index]
    IR = room.rir 

    # Extracting bright and dark zone mic coordinates for convenience
    bright_zone_mics = [mic_positions_list[i] for i in bright_zone_mics_index]
    dark_zone_mics = [mic_positions_list[i] for i in dark_zone_mics_index]

    return room, IR, mic_positions, sources_list, M_b, M_d, bright_zone_mics, dark_zone_mics

# --- Helper function to generate microphone grid ---

def _generate_mic_grid(room_dim):
    mic_positions_list = []
    bright_indices = []
    dark_indices = []

    idx = 0
    room_center_x = room_dim[0] / 2

    # Left half (Bright zone)
    for x_m in np.linspace(0.5, room_center_x - 0.5, 6):
        for y_m in np.linspace(0.5, room_dim[1] - 0.5, 4):
            mic_positions_list.append([x_m, y_m, 1.5])
            bright_indices.append(idx)
            idx += 1

    # Right half (Dark zone)
    for x_m in np.linspace(room_center_x + 0.5, room_dim[0] - 0.5, 6):
        for y_m in np.linspace(0.5, room_dim[1] - 0.5, 4):
            mic_positions_list.append([x_m, y_m, 1.5])
            dark_indices.append(idx)
            idx += 1

    return mic_positions_list, bright_indices, dark_indices

# --- Main Setup and Execution ---

# Generate inputs
demo_sources = [[i, 0, 1.5] for i in np.linspace(2, 4, 10)]
demo_mic_list, bright_zone_mics_index, dark_zone_mics_index = _generate_mic_grid(room_dim)

# --- Define Custom Directivity Parameters ---
# Example: Make the first mic in the bright zone (index 0) and the first mic
# in the dark zone (index 24) cardioid.

# Index 0 (Bright zone mic) - Pointing towards the center (x=3)
mic0_orientation = np.array([0, 0, 0]) # Yaw 0 (positive X-axis) if near x=0.5
# If the room center is x=3, and mic is near x=0.5, pointing x=3 is positive X (Yaw=0)

# Index 24 (Dark zone mic) - Pointing away from the center (x=3)
mic24_orientation = np.array([np.pi, 0, 0]) # Yaw 180 degrees (negative X-axis)
# If mic is near x=5.5, pointing away from center (x=3) is positive X (Yaw=0).
# Let's assume it points towards the bright zone for suppression, so towards -X (Yaw=180 deg)

demo_cardioid_indices = [0, 24]
demo_cardioid_orientations = [mic0_orientation, mic24_orientation]

# --- Call the parameterized function ---
print("--- Setting up Acoustic Scenario ---")
room, IR, mic_positions, sources, M_b, M_d, bright_zone_mics, dark_zone_mics = setup_acoustic_scenario( 
    sources=demo_sources, 
    mic_positions_list=demo_mic_list, 
    bright_zone_mics_index=bright_zone_mics_index, 
    dark_zone_mics_index=dark_zone_mics_index,
    fs_target=fs_target)

n_mics = mic_positions.shape[1]
n_srcs = len(sources)

# K is the maximum RIR length, which may vary slightly across mics/sources due to floating point math
K = max(len(IR[m][s]) for m in range(n_mics) for s in range(n_srcs))
print(f"Max RIR length K = {K}")

def build_HB_time_series(IR, bright_mics, N):
    """
    Construct H_B[n] with shape (N, M_B, L):
    - N = time samples
    - M_B = number of bright-zone microphones
    - L = number of loudspeakers

    H_B[n][m,l] = IR response sample at time n for (mic m, speaker l)
    """
    M_B = len(bright_mics)
    L = len(IR[0])  # number of sources/loudspeakers

    H_B = np.zeros((N, M_B, L), dtype=float)
    for n in range(N):
        for mi, m in enumerate(bright_mics):
            for l in range(L):
                H_B[n, mi, l] = IR[m][l][n]  # pick impulse response sample at time n

    return H_B

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

def compute_q_vast(V, mu, lambda_vals, U, r_B):
    q = np.zeros_like(r_B)
    for v in range(V):
        weight = lambda_vals[v] / (lambda_vals[v] + mu)
        projection = np.dot(U[:, v].T, r_B)
        q += weight * projection * U[:, v]
    return q

def compute_rB_sigma2(bright_mics, x, IR, d_B, N, J):
    LJ = n_srcs * J
    r_B = np.zeros((LJ,), dtype=float)
    sigma_d_sq = 0.0

    for mi, m in enumerate(bright_mics):
        U_m = build_Um_for_mic(m, x, IR, N, J)  # (N, LJ)
        d_vec = d_B[:, mi]                     # (N,)
        r_B += U_m.T @ d_vec                   # (LJ,)
        sigma_d_sq += np.sum(d_vec ** 2)

    r_B /= (len(bright_mics) * N)
    sigma_d_sq /= (len(dark_zone_mics) * N)

    return r_B, sigma_d_sq


print("Building R_B (bright) and R_D (dark). This may take some time...")

tstart = time.perf_counter()
R_B = build_R_from_micset(bright_zone_mics_index, x, IR, N, J)
R_D = build_R_from_micset(dark_zone_mics_index, x, IR, N, J)
print("Built R_B and R_D in {:.2f} s".format(time.perf_counter() - tstart))

print("R_B trace:", np.trace(R_B), "R_D trace:", np.trace(R_D))
# Regularize R_D slightly to ensure pos-definite for generalized eigenproblem
R_D_reg = R_D + reg_eps * np.eye(R_D.shape[0])

lambda_vals, U = eigh(R_B, R_D)  # generalized eigenvalue problem

# Sort eigenvalues descending
idx = np.argsort(-lambda_vals.real)
lambda_vals = lambda_vals.real[idx]
U = U[:, idx]

H_B = build_HB_time_series(IR, bright_zone_mics_index, N)

d_B = np.ones((N, len(bright_zone_mics)))*0.3536

r_B, sigma_d_sq = compute_rB_sigma2(bright_zone_mics_index, x, IR, d_B, N, J)

q_vec = compute_q_vast(V, mu, lambda_vals, U, r_B)

q_matrix = q_vec.reshape(n_srcs, J)

def compute_SB_SD(V, mu, lambda_vals, U, r_B, sigma_d_sq):
    SB = sigma_d_sq
    SD = 0
    for v in range(V):
        proj_norm_sq = np.abs(np.dot(U[:, v].T, r_B))**2
        SB -= (lambda_vals[v]**2 / (lambda_vals[v] + mu)**2) * proj_norm_sq
        SD += (1 / (lambda_vals[v] + mu)**2) * proj_norm_sq
    return SB, SD

SB, SD = compute_SB_SD(V, mu, lambda_vals, U, r_B, sigma_d_sq)

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

#print("Computing pressure field (coarse grid for speed)...")
# Use a short segment of the signal for visualization to speed up convolution
test_signal = wav[:fs_target//4] if len(wav) >= fs_target//4 else wav
tstart = time.perf_counter()
# Note: we pass the sources list and fs_target to the function for direct path calculation
Xg, Yg, P = pressure_field_from_q(q_matrix, IR, test_signal, sources, room_dim, fs_target, grid_res=grid_res, z_plane=z_plane)
#print("Pressure computed in {:.2f}s".format(time.perf_counter() - tstart))

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