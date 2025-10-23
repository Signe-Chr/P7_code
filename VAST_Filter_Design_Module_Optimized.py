import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import time
import os
import torch
import torch.fft as tfft

# ---------------------------
# Helpers: device selection
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)

# ---------------------------
# Room setup (same as before)
# ---------------------------
def setup_acoustic_scenario(sources, mic_positions_list, bright_zone_mics_index,
                            dark_zone_mics_index, fs_target, room_dim, rt60,
                            mic_directions, user_rotation):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs_target, materials=pra.Material(e_absorption), max_order=max_order)
    for s in sources:
        room.add_source(s)
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(mic_positions, room.fs)
    room.add_microphone_array(mic_array)
    try:
        room.mic_array.set_directivity(
            mic_directions[:-1] + [pra.directivities.HyperCardioid(
                pra.directivities.DirectionVector(user_rotation - np.pi/2)
            )]
        )
    except Exception:
        pass
    print(f"Computing RIRs for {mic_positions.shape[1]} mics and {len(sources)} sources...")
    room.compute_rir()
    IR = room.rir
    return IR, len(bright_zone_mics_index), len(dark_zone_mics_index)

# gpu_vast.py
import numpy as np
import time

try:
    import cupy as cp
    xp = cp
    gpu_available = True
except Exception:
    cp = None
    xp = np
    gpu_available = False

from scipy.linalg import toeplitz  # keep for fallback; we will reimplement with xp for GPU case
from scipy.linalg import eigh as np_eigh  # fallback eigh
# Note: if using CuPy, use cp.linalg.eigh

# Utility: ensure arrays stay on correct device
def as_xp_array(arr, dtype=xp.float32):
    if xp is np:
        return np.asarray(arr, dtype=dtype)
    else:
        return cp.asarray(arr, dtype=dtype)

# --- GPU/NumPy-aware U builder ---
def build_U_l_single_gpu(x, N, J, dtype=xp.float32):
    """
    Build U^l (N x J) using sliding window view on xp (np or cupy).
    For NumPy we use numpy.lib.stride_tricks.sliding_window_view; for CuPy we use cupy.lib.stride_tricks.sliding_window_view (available).
    """
    x_vec = x[:N].copy() if len(x) >= N else np.pad(x, (0, N - len(x)))
    x_vec = as_xp_array(x_vec, dtype=dtype)  # moves to device if cp

    # We want a Toeplitz where columns are x_vec shifted (causal): column j = x_vec shifted by j (zero-padded)
    # Equivalent to building a sliding window of length J on a zero-padded x_vec
    pad = xp.zeros(J - 1, dtype=dtype)
    x_padded = xp.concatenate([x_vec, pad])  # length N + J -1

    # sliding_window_view produces shape (N, J) where row i = x_padded[i:i+J]
    try:
        from numpy.lib.stride_tricks import sliding_window_view as np_swv
        # CuPy also has sliding_window_view under xp.lib for newer versions
        swv = xp.lib.stride_tricks.sliding_window_view
    except Exception:
        # Fallback: build with explicit indexing (less efficient)
        Np = x_padded.shape[0] - J + 1
        U = xp.empty((N, J), dtype=dtype)
        for j in range(J):
            U[:, j] = x_padded[j:j+N]
        return U

    U = swv(x_padded, window_shape=J)[:N, :]  # shape (N, J)
    return U

# --- Build R_UU template ---
def build_R_template_gpu(x, N, J, n_srcs, dtype=xp.float32):
    """
    Returns R_UU (LJ x LJ) and U_l_blocks (list of (N,J) xp arrays).
    All on device if xp is cupy.
    """
    # build U_l blocks (identical for all sources if same x), but we still store references
    U_single = build_U_l_single_gpu(x, N, J, dtype=dtype)  # (N, J)
    # If n_srcs is large and U_single small, repeating horizontally is cheap:
    U_l_blocks = [U_single] * n_srcs  # share same array (no copy)
    U_m_template = xp.hstack([U_single for _ in range(n_srcs)])  # (N, LJ)
    # Compute Gram matrix on device
    R_UU = U_m_template.T @ U_m_template   # (LJ, LJ)
    return R_UU, U_l_blocks

# --- R builder (simple) ---
def build_R_from_template_gpu(R_UU, mic_indices, J, n_srcs, reg_eps=0):
    LJ = n_srcs * J
    R = R_UU / max(1, len(mic_indices))
    if reg_eps > 0:
        R = R + reg_eps * xp.eye(LJ, dtype=R.dtype)
    return R

# --- r_B on GPU: do one matmul over all bright mics at once ---
def compute_rB_gpu(bright_mics_index, d_B, N, J, n_srcs, U_l_blocks):
    """
    Compute r_B. Assumes d_B is (N, M_b) as NumPy or CuPy array.
    We try to perform: r_B = (U_m_template.T @ D).sum(axis=1) / (M_b * N)
    Equivalent to r_B = U_m_template.T @ (sum over m of d_B[:,m]) / (M_b * N)
    """
    U_single = U_l_blocks[0]  # xp array shape (N,J)
    U_m_template = xp.hstack([U_single for _ in range(n_srcs)])  # (N, LJ)
    # Move d_B to xp
    d_B_xp = as_xp_array(d_B, dtype=U_single.dtype)  # (N, M_b)
    # Sum across mics to get (N,) vector
    s = d_B_xp.sum(axis=1)  # (N,)
    r_B = U_m_template.T @ s  # (LJ,)
    denom = (d_B_xp.shape[1] * N)
    r_B = r_B / denom
    return r_B

# --- compute q using matrix ops (no loop) ---
def compute_q_vast_gpu(V, mu, lambda_vals, U, r_B):
    """
    Use matrix formulation:
      For selected V eigenvectors U_V (LJ x V) and lambda_vals_V (V,)
      weight diag = lambda / (lambda + mu)
      q = U_V @ ( diag(weight) @ (U_V.T @ r_B) )
    This avoids per-eigenvector Python loops.
    """
    # Convert to xp arrays
    lambda_vals = xp.asarray(xp.real(lambda_vals), dtype=r_B.dtype)
    U = xp.asarray(xp.real(U), dtype=r_B.dtype)
    r_B = xp.asarray(xp.real(r_B), dtype=r_B.dtype)

    V = min(V, U.shape[1])
    U_V = U[:, :V]               # (LJ, V)
    lambda_V = lambda_vals[:V]   # (V,)

    # compute intermediate: alpha = U_V.T @ r_B  => shape (V,)
    alpha = U_V.T @ r_B          # (V,)
    weights = lambda_V / (lambda_V + mu)  # (V,)
    # weighted alpha
    alpha_w = weights * alpha    # (V,)

    q = U_V @ alpha_w            # (LJ,)
    return q

# --- eigen solver wrapper ---
def generalized_eigh_gpu(R_B, R_D):
    """
    Solve generalized eigenproblem R_B x = lambda R_D x on xp (CuPy) if available,
    otherwise fall back to scipy.linalg.eigh on CPU (converted back to np arrays).
    """
    if gpu_available:
        # CuPy supports generalized eigen via cupy.linalg.eigh for generalized? 
        # If not available, convert to dense standard problem: solve R_D^{-1} R_B
        # But better: use cupy.linalg.eigh(R_B, R_D) if provided.
        try:
            lambda_vals, U = cp.linalg.eigh(R_B, R_D)
            return lambda_vals, U
        except Exception:
            # fallback: compute R_D^{-1} @ R_B on device (careful about stability)
            # Use cholesky of R_D: R_D = L L^H, solve L^{-1} R_B L^{-T} etc.
            L = cp.linalg.cholesky(R_D)
            invL = cp.linalg.inv(L)
            C = invL @ R_B @ invL.T.conj()
            lambda_vals, W = cp.linalg.eigh(C)
            # transform eigenvectors back
            U = invL.T.conj() @ W
            return lambda_vals, U
    else:
        # CPU fallback: convert back to numpy
        R_B_np = np.asarray(R_B)
        R_D_np = np.asarray(R_D)
        lambda_vals, U = np_eigh(R_B_np, R_D_np)
        return lambda_vals, U

# --- Top-level design function that uses GPU functions ---
def design_vast_filter1(sources, mic_positions_list,
                           bright_zone_mics_index, dark_zone_mics_index,
                           x, rt60, direction_list, user_rotation, fs_target, J, N,
                           V, mu, room_dim, reg_eps, target_amplitude,
                           dtype=xp.float32):
    # Keep RIR generation on CPU (pyroomacoustics) because pyroomacoustics is CPU-only
    # --- you can call your existing setup_acoustic_scenario to get IR on CPU ---
    IR = setup_acoustic_scenario(sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, fs_target, room_dim, rt60, direction_list, user_rotation)
    #sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, fs_target, room_dim, rt60, mic_directions, user_rotation

    n_srcs = len(sources)
    M_b = len(bright_zone_mics_index)

    t0 = time.perf_counter()
    R_UU, U_l_blocks = build_R_template_gpu(x, N, J, n_srcs, dtype=dtype)
    t1 = time.perf_counter()

    R_B = build_R_from_template_gpu(R_UU, bright_zone_mics_index, J, n_srcs, reg_eps=0)
    R_D = build_R_from_template_gpu(R_UU, dark_zone_mics_index, J, n_srcs, reg_eps=reg_eps)

    # Solve GEP
    lambda_vals, U = generalized_eigh_gpu(R_B, R_D)

    # Sort descending
    if gpu_available:
        idx = cp.argsort(-lambda_vals.real)
        lambda_vals = lambda_vals.real[idx]
        U = U[:, idx]
    else:
        idx = np.argsort(-lambda_vals.real)
        lambda_vals = lambda_vals.real[idx]
        U = U[:, idx]

    V = min(V, len(lambda_vals))

    # desired signal
    d_B = np.ones((N, M_b), dtype=np.float32) * target_amplitude

    r_B = compute_rB_gpu(bright_zone_mics_index, d_B, N, J, n_srcs, U_l_blocks)

    # compute q
    q_vec = compute_q_vast_gpu(V, mu, lambda_vals, U, r_B)

    # bring q back to host if on GPU
    if gpu_available:
        q_vec = cp.asnumpy(q_vec)

    q_matrix = q_vec.reshape(n_srcs, J)
    return q_matrix, IR
