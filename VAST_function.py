import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz, eigh
import matplotlib.pyplot as plt
import time
import os
from Junk import Room_configuration as rc

'''
def setup_acoustic_scenario(sources, 
                            mic_positions_list, 
                            bright_zone_mics_index, 
                            dark_zone_mics_index, 
                            fs_target, 
                            room_dim, 
                            absorption, 
                            max_order):
    """
    Sets up a pyroomacoustics simulation environment (ShoeBox) and computes RIRs.
    
    Returns:
        tuple: (IR, M_b, M_d)
            IR (list of lists): Room Impulse Responses.
            M_b (int): Number of bright zone microphones.
            M_d (int): Number of dark zone microphones.
    """
    sources_list = sources 

    M_b, M_d = len(bright_zone_mics_index), len(dark_zone_mics_index)
    
    # Define Room
    room = pra.ShoeBox(
        room_dim,
        fs=fs_target,
        materials=pra.Material(absorption),
        max_order=max_order,
    )

    # Add Sources (Loudspeakers)
    for s in sources_list:
        room.add_source(s)

    # Define and Add Microphone Grid
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(
        mic_positions,
        room.fs)
    room.add_microphone_array(mic_array)

    # Compute RIRs
    print(f"Computing RIRs for {mic_positions.shape[1]} mics (Bright: {M_b}, Dark: {M_d}) and {len(sources_list)} sources...")
    room.compute_rir()

    # RIRs are stored in room.rir: room.rir[mic_index][source_index]
    IR = room.rir

    room.sources = []
    room.mic_array = None
    
    return IR, M_b, M_d

def build_U_ml_single(x, h_ml, N, J):
    """ Build U^{m,l} (N x J) for a single mic m and single speaker l (Toeplitz matrix). """
    # u = x * h_ml (full conv), truncated to N samples
    u = fftconvolve(x, h_ml)[:N]
    if u.shape[0] < N:
        u = np.pad(u, (0, N - u.shape[0]))
    
    # Create the Toeplitz matrix (first column is u, first row is zeros of length J)
    first_col = u
    first_row = np.zeros(J)
    U_ml = toeplitz(first_col, first_row)[:N, :J]
    return U_ml

def build_Um_for_mic(m_idx, x, IR, N, J, n_srcs):
    """ 
    Build U^m for microphone index m_idx by horizontally concatenating U^{m,l} for all l.
    Returns U_m (N x (L*J)).
    """
    U_blocks = []
    for l in range(n_srcs):
        h_ml = IR[m_idx][l]
        U_ml = build_U_ml_single(x, h_ml, N, J)
        U_blocks.append(U_ml)
    U_m = np.hstack(U_blocks)
    return U_m

def build_R_from_micset(mic_indices, x, IR, N, J, n_srcs, reg_eps=0):
    """
    Compute R = (1/|M|) * sum_{m in M} U^m.T @ U^m 
    R is the average covariance approximation over the microphones in the set.
    """
    LJ = n_srcs * J
    R = np.zeros((LJ, LJ), dtype=float)
    
    for m in mic_indices:
        U_m = build_Um_for_mic(m, x, IR, N, J, n_srcs)
        R += U_m.T @ U_m
        
    R /= max(1, len(mic_indices))
    
    # Apply regularization if needed (for R_D)
    if reg_eps > 0:
        R += reg_eps * np.eye(LJ)
        
    return R
'''
    
def compute_rB(bright_mics_index, x, IR, d_B, N, J, n_srcs):
    """ Compute the cross-correlation vector r_B = E[U_B^T d_B] """
    LJ = n_srcs * J
    r_B = np.zeros((LJ,), dtype=float)

    for mi, m in enumerate(bright_mics_index):
        U_m = rc.build_Um_for_mic(m, x, IR, N, J)       # (N, LJ)
        d_vec = d_B[:, mi]                              # (N,)
        r_B += U_m.T @ d_vec                            # (LJ,)

    # Normalize by the total number of data points
    r_B /= (len(bright_mics_index) * N)

    return r_B


def compute_q_vast(V, mu, lambda_vals, U, r_B):
    """ Compute the VAST filter vector q """
    q = np.zeros_like(r_B)
    for v in range(V):
        weight = lambda_vals[v] / (lambda_vals[v] + mu)
        projection = np.dot(U[:, v].T, r_B)
        q += weight * projection * U[:, v]
    return q

# --- 2. MAIN DESIGN FUNCTION ---

def VAST_solution(sources, mic_positions_list,
                          wav_path, fs_target, J, N, 
                          V, mu, reg_eps, target_amplitude=0.3536):
    """
    Designs the VAST Time-Domain filter coefficients (q_matrix) and saves them.

    Args (Acoustic Setup):
        sources (list of lists): Coordinates of the sound sources (loudspeakers).
        mic_positions_list (list of lists): Coordinates of all microphones.
        bright_zone_mics_index (list): Indices (0-based) of mics in the bright zone.
        dark_zone_mics_index (list): Indices (0-based) of mics in the dark zone.
        
    Args (Parameters):
        wav_path (str): Path to the excitation .wav file.
        out_q_path (str): Path to save the resulting q_matrix (.npy).
        fs_target (int): Sampling rate for RIR simulation and signal processing.
        J (int): Filter length (number of taps) per loudspeaker.
        N (int): Number of time samples used for covariance matrix estimation.
        V (int): Number of dominant eigenmodes to use in the VAST solution.
        mu (float): Regularization parameter for VAST.
        room_dim (list): [L, W, H] dimensions of the room (meters).
        absorption (float): Wall absorption coefficient for RIR simulation.
        max_order (int): Maximum reflections for RIR computation.
        reg_eps (float): Regularization factor for R_D (dark zone covariance).
        target_amplitude (float): Constant amplitude for the desired bright zone signal d_B.

    Returns:
        np.ndarray: The resulting filter coefficient matrix q_matrix (L x J).
    """

    print("--- Starting VAST Time-Domain Filter Design ---")
    t_start_total = time.perf_counter()

    # --- Load and Prepare Signal (x) ---
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} not found - adjust path.")
    fs_wav, wav = wavfile.read(wav_path)

    
    if fs_wav != fs_target:
        print(f"Warning: wav sample rate {fs_wav} != target {fs_target}. This implementation does not resample.")

    wav = np.array(wav, dtype=float)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav / (np.max(np.abs(wav)) + 1e-12)
    x = wav[:N].copy() if len(wav) >= N else np.pad(wav, (0, max(0, N-len(wav))), mode='constant')

    # --- Setup Acoustic Scenario and Compute RIRs ---
    n_srcs = len(sources)
    n_mics = len(mic_positions_list)
    
    IR = rc.IR
    M_b = len(rc.bright_zone_mics)
    M_d = len(rc.dark_zone_mics)
    #IR, M_b, M_d = setup_acoustic_scenario(
    #    sources=sources, 
    #    mic_positions_list=mic_positions_list, 
    #    bright_zone_mics_index=bright_zone_mics_index, 
    #    dark_zone_mics_index=dark_zone_mics_index,
    #    fs_target=fs_target, room_dim=room_dim, absorption=absorption, max_order=max_order)
    
    # --- Compute Covariance Matrices R_B and R_D ---
    R_D, R_B, R_D_reg, r_d = rc.build_R()

    # --- Solve Generalized Eigenvalue Problem (GEP) ---
    print("Solving Generalized Eigenvalue Problem...")
    lambda_vals, U = eigh(R_B, R_D)

    # Sort eigenvalues descending
    idx = np.argsort(-lambda_vals.real)
    lambda_vals = lambda_vals.real[idx]
    U = U[:, idx]
    
    # Check V (number of modes) vs total modes
    V = min(V, len(lambda_vals))
    print(f"Using V={V} modes for VAST solution.")

    # --- Compute VAST Filter Vector (q) ---
    
    # Define desired signal d_B (constant amplitude across time/mics)
    d_B = np.ones((N, M_b)) * target_amplitude
    r_B = compute_rB(rc.bright_zone_mics, x, IR, d_B, N, J, n_srcs)
    q_vec = compute_q_vast(V, mu, lambda_vals, U, r_B)
    
    # Reshape q vector into a matrix (Loudspeakers x Taps)
    q_matrix = q_vec.reshape(n_srcs, J)

    # --- Save Coefficients ---
    #np.save(out_q_path, q_matrix)
    #print(f"Successfully designed filter and saved q_matrix to {out_q_path} in {time.perf_counter() - t_start_total:.2f} s")
    return q_vec, q_matrix

if __name__ == "__main__":
    q_vec, q_matrix = VAST_solution(
        sources=rc.sources,
        mic_positions_list=rc.mic_positions_list,
        wav_path=rc.wav_path,
        fs_target=rc.fs_target,
        J=rc.J,
        N=rc.N,
        V=rc.V,
        mu=rc.mu,
        reg_eps=rc.reg_eps,
        target_amplitude=rc.target_amplitude
    )
    print(q_vec, q_matrix)

