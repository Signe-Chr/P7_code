import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz, eigh
import time
import os
import matplotlib.pyplot as plt
import time
from numba import njit, prange


def setup_acoustic_scenario(sources, 
                        mic_positions_list, 
                        bright_zone_mics_index, 
                        dark_zone_mics_index, 
                        fs_target, 
                        room_dim, 
                        rt60,
                        mic_directions, 
                        user_rotation):
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
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs_target,
        materials=pra.Material(e_absorption),
        max_order=max_order)
    
    # Add Sources (Loudspeakers)
    for s in sources_list:
        room.add_source(s)

    # Define and Add Microphone Grid
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(
        mic_positions,
        room.fs)
    room.add_microphone_array(mic_array)

    room.mic_array.set_directivity(mic_directions[:-1]+[pra.directivities.HyperCardioid(
                    pra.directivities.DirectionVector(user_rotation-np.pi/2)
            )])

    # Compute RIRs
    #print(f"Computing RIRs for {mic_positions.shape[1]} mics (Bright: {M_b}, Dark: {M_d}) and {len(sources_list)} sources...")
    room.compute_rir()



    # RIRs are stored in room.rir: room.rir[mic_index][source_index]
    IR = room.rir 

    return IR, M_b, M_d

@njit(parallel=False, fastmath=True)
def toeplitz_block_numba(cross, center, J):
    """Fast Toeplitz block builder using Numba instead of scipy.linalg.toeplitz.
       cross: full cross-correlation vector (len = 2N-1)
       center: index of zero-lag (N-1)
       J: number of filter taps
    """
    out = np.empty((J, J), dtype=np.float64)
    for i in range(J):
        for j in range(J):
            out[i, j] = cross[center + i - j]
    return out

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
    """Compute R = (1/|M|) * sum_{m in M} U^m.T @ U^m, optimized with Numba Toeplitz blocks."""
    LJ = n_srcs * J
    R = np.zeros((LJ, LJ), dtype=np.float64)
    center = N - 1

    for m in mic_indices:
        # Compute all u_{m,l}(t)
        u_list = [fftconvolve(x, IR[m][l])[:N] for l in range(n_srcs)]
        Rm = np.zeros((LJ, LJ), dtype=np.float64)

        for l in range(n_srcs):
            u_l = u_list[l]
            for k in range(l, n_srcs):
                u_k = u_list[k]
                cross = fftconvolve(u_l, u_k[::-1])  # cross-correlation
                block = toeplitz_block_numba(cross, center, J)
                i0 = l * J
                j0 = k * J
                Rm[i0:i0 + J, j0:j0 + J] = block
                if k != l:
                    Rm[j0:j0 + J, i0:i0 + J] = block.T
        R += Rm

    R /= max(1, len(mic_indices))

    if reg_eps > 0:
        R += reg_eps * np.eye(LJ)

    return R

@njit(fastmath=True)
def compute_rB_numba(bright_mics_index, x, IR_list, d_B, N, J, n_srcs):
    """Fast version of r_B computation using direct convolution and Numba."""
    LJ = n_srcs * J
    r_B = np.zeros(LJ, dtype=np.float64)
    center = N - 1

    for mi in range(len(bright_mics_index)):
        m = bright_mics_index[mi]
        d_vec = d_B[:, mi]
        for l in range(n_srcs):
            h_ml = IR_list[m][l]
            u_ml = np.convolve(x, h_ml)[:N]
            # Cross-correlate d_B with u_ml (equivalent to U_ml.T @ d_B)
            cross = np.convolve(u_ml, d_vec[::-1])
            start = center
            r_seg = np.zeros(J)
            for j in range(J):
                r_seg[j] = cross[start + j]
            r_B[l * J : (l + 1) * J] += r_seg

    r_B /= (len(bright_mics_index) * N)
    return r_B

def compute_rB(bright_mics_index, x, IR, d_B, N, J, n_srcs):
    """ Compute the cross-correlation vector r_B = E[U_B^T d_B] """
    LJ = n_srcs * J
    r_B = np.zeros((LJ,), dtype=float)

    for mi, m in enumerate(bright_mics_index):
        U_m = build_Um_for_mic(m, x, IR, N, J, n_srcs)  # (N, LJ)
        d_vec = d_B[:, mi]                              # (N,)
        r_B += U_m.T @ d_vec                            # (LJ,)

    # Normalize by the total number of data points
    r_B /= (len(bright_mics_index) * N)

    return r_B


def compute_q_vast(V, mu, lambda_vals, U, r_B):
    q = np.zeros_like(r_B)
    for v in range(V):
        weight = lambda_vals[v] / (lambda_vals[v] + mu)
        projection = np.dot(U[:, v].T, r_B)
        q += weight * projection * U[:, v]
    return q


def prepare_rir_input(IR, n_mics, n_srcs, max_length=512):
    rir_list = []

    for mic_idx in range(n_mics):
        rir_temp = []
        for src_idx in range(n_srcs):
            rir = IR[mic_idx][src_idx]
            # Truncate or zero-pad to max_length
            if len(rir) > max_length:
                rir = rir[:max_length]
            else:
                rir = np.pad(rir, (0, max_length - len(rir)))
            rir_temp.append(rir)
        rir_list.append(rir_temp)

    
    return np.array(rir_list)


def design_vast_filter(sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
<<<<<<< HEAD
                        wav_path, rt60, direction_list, user_rotation, fs_target, J, N, 
=======
                        wav, rt60, direction_list, user_rotation, fs_target, J, N, 
>>>>>>> 2479a2105e19706b89072158bd4476b22faadf06
                        V, mu, room_dim, reg_eps, target_amplitude):

    #print("--- Starting VAST Time-Domain Filter Design ---")
    t_start_total = time.perf_counter()

<<<<<<< HEAD
    # --- Load and Prepare Signal (x) ---
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} not found - adjust path.")
    fs_wav, wav = wavfile.read(wav_path)


    if fs_wav != fs_target:
        pass
        #print(f"Warning: wav sample rate {fs_wav} != target {fs_target}. This implementation does not resample.")
=======
>>>>>>> 2479a2105e19706b89072158bd4476b22faadf06

    wav = np.array(wav, dtype=float)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav / (np.max(np.abs(wav)) + 1e-12)
    x = wav[:N].copy() if len(wav) >= N else np.pad(wav, (0, max(0, N-len(wav))), mode='constant')

    # --- Setup Acoustic Scenario and Compute RIRs ---
    n_srcs = len(sources)
    n_mics = len(mic_positions_list)

    IR, M_b, M_d = setup_acoustic_scenario(
        sources=sources, 
        mic_positions_list=mic_positions_list, 
        bright_zone_mics_index=bright_zone_mics_index, 
        dark_zone_mics_index=dark_zone_mics_index,
        fs_target=fs_target, room_dim=room_dim, rt60=rt60, mic_directions=direction_list, user_rotation=user_rotation
    )

    # --- Compute Covariance Matrices R_B and R_D ---
    tstart = time.perf_counter()
    R_B = build_R_from_micset(bright_zone_mics_index, x, IR, N, J, n_srcs, reg_eps=0)
    R_D = build_R_from_micset(dark_zone_mics_index, x, IR, N, J, n_srcs, reg_eps=reg_eps)
    #print("Built R_B and R_D in {:.2f} s".format(time.perf_counter() - tstart))

    # --- Solve Generalized Eigenvalue Problem (GEP) ---
    #print("Solving Generalized Eigenvalue Problem...")
    LJ = R_B.shape[0]
    V_use = min(V, LJ)
    # compute only the largest V eigenpairs of the generalized problem
    # eigh returns ascending eigenvalues; subset_by_index=(LJ-V_use, LJ-1)
    lambda_vals, U = eigh(R_B, R_D, subset_by_index=(LJ - V_use, LJ - 1))
    # Sort eigenvalues descending
    idx = np.argsort(-lambda_vals.real)
    lambda_vals = lambda_vals.real[idx]
    U = U[:, idx]

    # Check V (number of modes) vs total modes
    V = min(V, len(lambda_vals))
    #print(f"Using V={V} modes for VAST solution.")

    # --- Compute VAST Filter Vector (q) ---

    # Define desired signal d_B (constant amplitude across time/mics)
    d_B = np.ones((N, M_b)) * target_amplitude
    max_len = max(len(ir) for mic_ir in IR for ir in mic_ir)
    n_mics = len(IR)
    n_srcs = len(IR[0])

    # Pad each IR to the same length
    IR_array = np.zeros((n_mics, n_srcs, max_len), dtype=np.float32)
    for m in range(n_mics):
        for s in range(n_srcs):
            ir = IR[m][s]
            L = len(ir)
            IR_array[m, s, :L] = ir.astype(np.float32)

    # Replace the old IR
    IR = IR_array
    x = np.array(x, dtype=np.float32)

    r_B = compute_rB_numba(bright_zone_mics_index, x, IR, d_B, N, J, n_srcs)

    q_vec = compute_q_vast(V, mu, lambda_vals, U, r_B)

    # Reshape q vector into a matrix (Loudspeakers x Taps)
    q_matrix = q_vec.reshape(n_srcs, J)

    # --- Save Coefficients ---
    #np.save(out_q_path, q_matrix)
    #print(f"Successfully designed filter and saved q_matrix to {out_q_path} in {time.perf_counter() - t_start_total:.2f} s")

    IR = prepare_rir_input(IR, n_mics, n_srcs, max_length=512)

    return q_matrix, IR

