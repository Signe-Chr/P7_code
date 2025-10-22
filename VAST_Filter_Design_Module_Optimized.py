import numpy as np
import pyroomacoustics as pra
from scipy.linalg import toeplitz, eigh
import time

# --- Acoustic Setup Functions ---

def setup_acoustic_scenario(sources, 
                            mic_positions_list, 
                            fs_target, 
                            room_dim, 
                            rt60,
                            mic_directions, 
                            user_rotation):
    """
    Sets up a pyroomacoustics simulation environment (ShoeBox) and computes RIRs.
    Returns: IR (list of lists)
    """
    # Define Room parameters
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs_target,
        materials=pra.Material(e_absorption),
        max_order=max_order)

    # Add Sources (Loudspeakers)
    for s in sources:
        room.add_source(s)

    # Define and Add Microphone Grid
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(
        mic_positions,
        room.fs)
    
    # Apply directivity: The last mic is the bright zone mic, rotated by user_rotation
    # The -np.pi/2 rotation aligns the primary axis with the user's forward view.
    mic_array.set_directivity(mic_directions[:-1] + [
        pra.directivities.HyperCardioid(
            pra.directivities.DirectionVector(user_rotation - np.pi/2)
        )
    ])
    
    room.add_microphone_array(mic_array)

    # Compute RIRs
    room.compute_rir()

    # RIRs are stored in room.rir: room.rir[mic_index][source_index]
    return room.rir

# --- VAST Matrix Construction Functions (Optimized for Reuse) ---

def build_U_l_single(x, N, J):
    """ 
    Build U^l (N x J) for a single speaker l (Toeplitz matrix) from the input signal x. 
    This is independent of the RIRs or microphones.
    """
    # Truncate/pad x to N samples
    x_vec = x[:N].copy() if len(x) >= N else np.pad(x, (0, N - len(x)))

    # The first column is x_vec, the first row are zeros (for causal filter)
    first_col = x_vec
    first_row = np.zeros(J)
    
    return toeplitz(first_col, first_row)[:N, :J]

def build_R_template(x, N, J, n_srcs):
    """
    Optimization: Computes the common U^T @ U term only once.
    Returns: 
      - R_UU (LJ x LJ): The core U.T @ U matrix.
      - U_l_blocks (list of N x J matrices): The individual U^l blocks.
    """
    
    # 1. Build U^l blocks
    U_l_blocks = [build_U_l_single(x, N, J) for _ in range(n_srcs)]

    # 2. Assemble U_m_template = [U^1 | U^2 | ... | U^L]
    U_m_template = np.hstack(U_l_blocks)
    
    # 3. Compute the core quadratic term
    R_UU = U_m_template.T @ U_m_template

    return R_UU, U_l_blocks

def build_R_from_template(R_UU, mic_indices, J, n_srcs, reg_eps=0):
    """
    Uses the pre-calculated R_UU template to compute R_B or R_D.
    """
    LJ = n_srcs * J
    
    # R is the average over the microphone set
    R = R_UU / max(1, len(mic_indices))

    # Apply regularization if needed (typically only for R_D)
    if reg_eps > 0:
        R += reg_eps * np.eye(LJ)

    return R

def compute_rB(bright_mics_index, d_B, N, J, n_srcs, U_l_blocks):
    """ Compute the cross-correlation vector r_B = E[U^T d] """
    LJ = n_srcs * J
    r_B = np.zeros((LJ,), dtype=float)

    # U^m is the same template for all mics in the bright zone
    U_m_template = np.hstack(U_l_blocks) # (N, LJ)

    for mi, m in enumerate(bright_mics_index):
        d_vec = d_B[:, mi]              # (N,)
        
        # r_B += U_m_template.T @ d_vec (The cross-correlation term)
        r_B += U_m_template.T @ d_vec

    # Normalize by the total number of data points (M_b * N)
    r_B /= (len(bright_mics_index) * N)

    return r_B

def compute_q_vast(V, mu, lambda_vals, U, r_B):
    """ Computes the VAST filter vector q using the V dominant modes (eigenvectors). """
    q = np.zeros_like(r_B, dtype=float)
    
    # Ensure components are real
    lambda_vals = np.real(lambda_vals)
    U = np.real(U)
    r_B = np.real(r_B)
    
    for v in range(V):
        # Regularized weighting factor
        weight = lambda_vals[v] / (lambda_vals[v] + mu)
        
        # Projection onto the v-th eigenvector
        projection = np.dot(U[:, v], r_B) 
        
        # Accumulate the weighted projection
        q += weight * projection * U[:, v]
    return q

# --- Main Design Function ---

def design_vast_filter(sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                        x, rt60, direction_list, user_rotation, fs_target, J, N, 
                        V, mu, room_dim, reg_eps, target_amplitude):

    n_srcs = len(sources)
    M_b = len(bright_zone_mics_index)
    
    # 1. Setup Acoustic Scenario and Compute RIRs
    IR = setup_acoustic_scenario(
        sources=sources, 
        mic_positions_list=mic_positions_list, 
        fs_target=fs_target, room_dim=room_dim, rt60=rt60, mic_directions=direction_list, user_rotation=user_rotation
    )

    # 2. Compute Covariance Matrix Templates
    tstart = time.perf_counter()
    R_UU, U_l_blocks = build_R_template(x, N, J, n_srcs)
    
    # 3. Compute Covariance Matrices R_B and R_D
    R_B = build_R_from_template(R_UU, bright_zone_mics_index, J, n_srcs, reg_eps=0)
    R_D = build_R_from_template(R_UU, dark_zone_mics_index, J, n_srcs, reg_eps=reg_eps)

    # 4. Solve Generalized Eigenvalue Problem (GEP)
    lambda_vals, U = eigh(R_B, R_D)

    # Sort eigenvalues descending (most important modes first)
    idx = np.argsort(-lambda_vals.real)
    lambda_vals = lambda_vals.real[idx]
    U = U[:, idx]

    # Check V (number of modes) vs total modes
    V = min(V, len(lambda_vals))

    # 5. Compute VAST Filter Vector (q)
    
    # Define desired signal d_B (constant amplitude across time/mics)
    d_B = np.ones((N, M_b)) * target_amplitude

    # Compute the cross-correlation term r_B
    r_B = compute_rB(bright_zone_mics_index, d_B, N, J, n_srcs, U_l_blocks)

    # Compute the final VAST filter vector q
    q_vec = compute_q_vast(V, mu, lambda_vals, U, r_B)

    # Reshape q vector into a matrix (Loudspeakers x Taps)
    q_matrix = q_vec.reshape(n_srcs, J)

    # Return both the filter matrix and the RIRs (as requested by the main script)
    return q_matrix, IR
