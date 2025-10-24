import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz, eigh
import time
import os
import matplotlib.pyplot as plt
import time


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


def design_vast_filter(
    sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
    wav, rt60, direction_list, user_rotation, fs_target, J, N, 
    V, mu, room_dim, reg_eps, target_amplitude
):
    print("\n--- Starting VAST Time-Domain Filter Design ---")
    t_start_total = time.perf_counter()

    # --- Preprocessing ---
    t0 = time.perf_counter()
    wav = np.array(wav, dtype=float)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav / (np.max(np.abs(wav)) + 1e-12)
    x = wav[:N].copy() if len(wav) >= N else np.pad(wav, (0, max(0, N-len(wav))), mode='constant')
    print(f"[{time.perf_counter() - t0:8.3f}s] Preprocessed wav")

    # --- Setup Acoustic Scenario and Compute RIRs ---
    t1 = time.perf_counter()
    IR, M_b, M_d = setup_acoustic_scenario(
        sources=sources, 
        mic_positions_list=mic_positions_list, 
        bright_zone_mics_index=bright_zone_mics_index, 
        dark_zone_mics_index=dark_zone_mics_index,
        fs_target=fs_target, room_dim=room_dim, rt60=rt60, 
        mic_directions=direction_list, user_rotation=user_rotation
    )
    print(f"[{time.perf_counter() - t1:8.3f}s] setup_acoustic_scenario() done")

    # --- Compute Covariance Matrices ---
    t2 = time.perf_counter()
    R_B = build_R_from_micset(bright_zone_mics_index, x, IR, N, J, len(sources), reg_eps=0)
    print(f"[{time.perf_counter() - t2:8.3f}s] build_R_from_micset() -> R_B done")

    t3 = time.perf_counter()
    R_D = build_R_from_micset(dark_zone_mics_index, x, IR, N, J, len(sources), reg_eps=reg_eps)
    print(f"[{time.perf_counter() - t3:8.3f}s] build_R_from_micset() -> R_D done")

    # --- Solve Generalized Eigenvalue Problem ---
    t4 = time.perf_counter()

    #print(np.shape(R_B), np.shape(R_D))
    lambda_vals, U = eigh(R_B, R_D)
    print(f"[{time.perf_counter() - t4:8.3f}s] eigh() done")

    # Sort eigenvalues
    idx = np.argsort(-lambda_vals.real)
    lambda_vals = lambda_vals.real[idx]
    U = U[:, idx]
    V = min(V, len(lambda_vals))
    print(f"[{time.perf_counter() - t_start_total:8.3f}s] Eigenvalues sorted (V={V})")

    # --- Compute VAST Filter Vector ---
    t5 = time.perf_counter()
    d_B = np.ones((N, M_b)) * target_amplitude
    r_B = compute_rB(bright_zone_mics_index, x, IR, d_B, N, J, len(sources))
    print(f"[{time.perf_counter() - t5:8.3f}s] compute_rB() done")

    t6 = time.perf_counter()
    q_vec = compute_q_vast(V, mu, lambda_vals, U, r_B)
    print(f"[{time.perf_counter() - t6:8.3f}s] compute_q_vast() done")

    q_matrix = q_vec.reshape(len(sources), J)

    # --- Prepare IR for output ---
    t7 = time.perf_counter()
    IR = prepare_rir_input(IR, len(mic_positions_list), len(sources), max_length=512)
    print(f"[{time.perf_counter() - t7:8.3f}s] prepare_rir_input() done")

    print(f"[{time.perf_counter() - t_start_total:8.3f}s] Total elapsed time")

    return q_matrix, IR